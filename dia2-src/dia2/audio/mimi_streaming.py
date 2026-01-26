"""
Streaming support for Mimi decoder.

This module patches the HuggingFace transformers Mimi implementation to support
stateful streaming decoding. The decoder's convolutional layers need to maintain
state across chunks to avoid boundary artifacts.

Usage:
    from dia2.audio.mimi_streaming import enable_streaming_decode, MimiDecoderPaddingCache

    # Enable streaming on a MimiModel instance
    enable_streaming_decode(model)

    # Create cache for streaming decode
    cache = MimiDecoderPaddingCache(model)

    # Decode with state
    audio1, cache = model.decode(codes1, decoder_padding_cache=cache, return_dict=False)
    audio2, cache = model.decode(codes2, decoder_padding_cache=cache, return_dict=False)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConvLayerCacheEntry:
    """Cache entry for a single convolutional layer."""
    padding_left: int
    padding_mode: str
    cached_states: Optional[torch.Tensor] = None


class MimiDecoderPaddingCache:
    """
    Padding cache for streaming Mimi decoder.

    Manages cached states for both regular Conv1d and ConvTranspose1d layers
    in the decoder to enable seamless chunk-by-chunk decoding.
    """

    def __init__(self, model: Any):
        """
        Initialize cache from a MimiModel instance.

        Args:
            model: A MimiModel instance (or its decoder attribute)
        """
        self.layer_caches: List[ConvLayerCacheEntry] = []
        self._initialized = False

        # Get decoder from model
        decoder = getattr(model, 'decoder', model)
        config = getattr(model, 'config', None) or getattr(decoder, 'config', None)

        if config is None:
            raise ValueError("Could not find config on model")

        # Analyze decoder layers and create cache entries
        self._analyze_decoder(decoder, config)

    def _analyze_decoder(self, decoder: nn.Module, config: Any):
        """Analyze decoder layers to determine caching requirements."""
        from transformers.models.mimi.modeling_mimi import (
            MimiConv1d,
            MimiConvTranspose1d,
            MimiResnetBlock,
        )

        layer_idx = 0
        for layer in decoder.layers:
            if isinstance(layer, MimiConv1d):
                # Regular conv needs left padding cache
                padding_left = getattr(layer, 'padding_left', 0)
                padding_total = getattr(layer, 'padding_total', padding_left)
                if layer.causal:
                    padding_left = padding_total
                self.layer_caches.append(ConvLayerCacheEntry(
                    padding_left=padding_left,
                    padding_mode='replicate' if config.pad_mode == 'replicate' else 'constant',
                ))
                layer._streaming_layer_idx = layer_idx
                layer_idx += 1

            elif isinstance(layer, MimiConvTranspose1d):
                # Transpose conv needs output tail cache
                # The amount to cache is padding_left (what gets trimmed from left)
                padding_left = getattr(layer, 'padding_left', 0)
                self.layer_caches.append(ConvLayerCacheEntry(
                    padding_left=padding_left,
                    padding_mode='constant',
                ))
                layer._streaming_layer_idx = layer_idx
                layer_idx += 1

            elif isinstance(layer, MimiResnetBlock):
                # ResnetBlock contains Conv1d layers that need caching
                for sublayer_name, sublayer in layer.named_modules():
                    if isinstance(sublayer, MimiConv1d):
                        padding_left = getattr(sublayer, 'padding_left', 0)
                        padding_total = getattr(sublayer, 'padding_total', padding_left)
                        if sublayer.causal:
                            padding_left = padding_total
                        self.layer_caches.append(ConvLayerCacheEntry(
                            padding_left=padding_left,
                            padding_mode='replicate' if config.pad_mode == 'replicate' else 'constant',
                        ))
                        sublayer._streaming_layer_idx = layer_idx
                        layer_idx += 1

    def get_cache(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get cached states for a layer."""
        if layer_idx < len(self.layer_caches):
            return self.layer_caches[layer_idx].cached_states
        return None

    def update_cache(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        is_transpose: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Update cache for a layer and return the padding to prepend.

        For regular Conv1d: cache input states, return cached input to prepend
        For ConvTranspose1d: cache output tail, return cached output to prepend

        Args:
            layer_idx: Index of the layer
            hidden_states: Current states (input for Conv1d, output for ConvTranspose1d)
            is_transpose: Whether this is a transpose convolution

        Returns:
            Cached states to prepend (or None if no caching needed)
        """
        if layer_idx >= len(self.layer_caches):
            return None

        entry = self.layer_caches[layer_idx]
        padding_left = entry.padding_left

        if padding_left == 0:
            return None

        batch_size = hidden_states.shape[0]
        channels = hidden_states.shape[1]
        seq_len = hidden_states.shape[2]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Get or initialize cache
        if entry.cached_states is None:
            # Initialize with zeros or replicated values
            if entry.padding_mode == 'replicate':
                entry.cached_states = hidden_states[:, :, :1].expand(-1, -1, padding_left).clone()
            else:
                entry.cached_states = torch.zeros(
                    batch_size, channels, padding_left,
                    device=device, dtype=dtype
                )

        # Get the cache to return (for prepending)
        cache_to_prepend = entry.cached_states.clone()

        # Update cache with new states (take the rightmost padding_left samples)
        if seq_len >= padding_left:
            entry.cached_states = hidden_states[:, :, -padding_left:].clone()
        else:
            # Not enough new states - combine old cache tail with new states
            shortfall = padding_left - seq_len
            entry.cached_states = torch.cat([
                entry.cached_states[:, :, -shortfall:],
                hidden_states
            ], dim=2)

        return cache_to_prepend

    def reset(self):
        """Reset all cached states."""
        for entry in self.layer_caches:
            entry.cached_states = None


def _patched_conv1d_forward(
    self,
    hidden_states: torch.Tensor,
    padding_cache: Optional[MimiDecoderPaddingCache] = None,
) -> torch.Tensor:
    """Patched forward for MimiConv1d with streaming support."""
    # Get extra padding needed
    extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

    layer_idx = getattr(self, '_streaming_layer_idx', None)

    if self.causal and padding_cache is not None and layer_idx is not None:
        # Use cached states instead of zero padding
        cached = padding_cache.update_cache(layer_idx, hidden_states, is_transpose=False)
        if cached is not None:
            hidden_states = torch.cat([cached, hidden_states], dim=2)
            # Only add extra_padding on right if needed
            if extra_padding > 0:
                hidden_states = self._pad1d(
                    hidden_states, (0, extra_padding), self.pad_mode
                )
        else:
            hidden_states = self._pad1d(
                hidden_states, (self.padding_total, extra_padding), self.pad_mode
            )
    elif self.causal:
        hidden_states = self._pad1d(
            hidden_states, (self.padding_total, extra_padding), self.pad_mode
        )
    else:
        hidden_states = self._pad1d(
            hidden_states,
            (self.padding_left, self.padding_right + extra_padding),
            self.pad_mode,
        )

    hidden_states = self.conv(hidden_states)
    return hidden_states


def _patched_conv_transpose1d_forward(
    self,
    hidden_states: torch.Tensor,
    padding_cache: Optional[MimiDecoderPaddingCache] = None,
) -> torch.Tensor:
    """Patched forward for MimiConvTranspose1d with streaming support."""
    layer_idx = getattr(self, '_streaming_layer_idx', None)

    # Run the transpose convolution
    hidden_states = self.conv(hidden_states)

    # Standard trimming
    end = hidden_states.shape[-1] - self.padding_right

    if padding_cache is not None and layer_idx is not None and self.causal:
        # Get cached output from previous chunk
        cached = padding_cache.get_cache(layer_idx)

        if cached is not None:
            # Prepend cached output (this fills in the "missing" left context)
            # The cached values complete the transient at the start of this chunk
            cache_len = cached.shape[-1]
            # Blend the cached tail with the new start (overlap-add style)
            if cache_len > 0 and self.padding_left > 0:
                blend_len = min(cache_len, self.padding_left, hidden_states.shape[-1])
                # Simple crossfade for smooth transition
                fade_in = torch.linspace(0, 1, blend_len, device=hidden_states.device, dtype=hidden_states.dtype)
                fade_out = 1 - fade_in
                hidden_states[:, :, :blend_len] = (
                    hidden_states[:, :, :blend_len] * fade_in +
                    cached[:, :, -blend_len:] * fade_out
                )

        # Update cache with the tail that would normally be trimmed
        # This is the "incomplete" output that depends on future input
        if end < hidden_states.shape[-1]:
            tail_to_cache = hidden_states[:, :, end:]
            padding_cache.update_cache(layer_idx, tail_to_cache, is_transpose=True)

    hidden_states = hidden_states[..., self.padding_left:end]
    return hidden_states


def _patched_resnet_block_forward(
    self,
    hidden_states: torch.Tensor,
    padding_cache: Optional[MimiDecoderPaddingCache] = None,
) -> torch.Tensor:
    """Patched forward for MimiResnetBlock with streaming support."""
    residual = hidden_states

    for layer in self.block:
        if hasattr(layer, '_streaming_layer_idx'):
            # This is a MimiConv1d that needs padding_cache
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        else:
            hidden_states = layer(hidden_states)

    if self.shortcut is not None:
        if hasattr(self.shortcut, '_streaming_layer_idx'):
            residual = self.shortcut(residual, padding_cache=padding_cache)
        else:
            residual = self.shortcut(residual)

    return residual + hidden_states


def _patched_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    padding_cache: Optional[MimiDecoderPaddingCache] = None,
) -> torch.Tensor:
    """Patched forward for MimiDecoder with streaming support."""
    from transformers.models.mimi.modeling_mimi import (
        MimiConv1d,
        MimiConvTranspose1d,
        MimiResnetBlock,
    )

    for layer in self.layers:
        if isinstance(layer, MimiConv1d):
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        elif isinstance(layer, MimiConvTranspose1d):
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        elif isinstance(layer, MimiResnetBlock):
            hidden_states = layer(hidden_states, padding_cache=padding_cache)
        else:
            hidden_states = layer(hidden_states)

    return hidden_states


def _patched_decode_frame(
    self,
    codes: torch.Tensor,
    decoder_past_key_values: Optional[Any] = None,
    decoder_padding_cache: Optional[MimiDecoderPaddingCache] = None,
    return_dict: Optional[bool] = None,
):
    """Patched _decode_frame with padding cache support."""
    embeddings = self.quantizer.decode(codes)
    embeddings = self.upsample(embeddings)

    decoder_outputs = self.decoder_transformer(
        embeddings.transpose(1, 2),
        past_key_values=decoder_past_key_values,
        return_dict=True,
    )
    decoder_past_key_values = decoder_outputs.past_key_values
    embeddings = decoder_outputs.last_hidden_state.transpose(1, 2)

    # Use patched decoder forward with padding cache
    outputs = self.decoder(embeddings, padding_cache=decoder_padding_cache)

    return outputs, decoder_past_key_values


def _patched_decode(
    self,
    audio_codes: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    decoder_past_key_values: Optional[Any] = None,
    decoder_padding_cache: Optional[MimiDecoderPaddingCache] = None,
    return_dict: Optional[bool] = None,
):
    """Patched decode method with padding cache support."""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    audio_values, decoder_past_key_values = self._decode_frame(
        audio_codes,
        decoder_past_key_values=decoder_past_key_values,
        decoder_padding_cache=decoder_padding_cache,
    )

    if not return_dict:
        return audio_values, decoder_past_key_values

    # Return with both caches
    from transformers.models.mimi.modeling_mimi import MimiDecoderOutput
    return MimiDecoderOutput(
        audio_values=audio_values,
        past_key_values=decoder_past_key_values,
    )


_original_methods = {}


def enable_streaming_decode(model: Any) -> None:
    """
    Enable streaming decode support on a MimiModel instance.

    This patches the model's methods to accept and use decoder_padding_cache
    for stateful streaming decoding.

    Args:
        model: A MimiModel instance
    """
    from transformers.models.mimi.modeling_mimi import (
        MimiConv1d,
        MimiConvTranspose1d,
        MimiResnetBlock,
        MimiDecoder,
        MimiModel,
    )

    # Patch Conv1d
    if MimiConv1d not in _original_methods:
        _original_methods[MimiConv1d] = MimiConv1d.forward
    MimiConv1d.forward = _patched_conv1d_forward

    # Patch ConvTranspose1d
    if MimiConvTranspose1d not in _original_methods:
        _original_methods[MimiConvTranspose1d] = MimiConvTranspose1d.forward
    MimiConvTranspose1d.forward = _patched_conv_transpose1d_forward

    # Patch ResnetBlock
    if MimiResnetBlock not in _original_methods:
        _original_methods[MimiResnetBlock] = MimiResnetBlock.forward
    MimiResnetBlock.forward = _patched_resnet_block_forward

    # Patch Decoder
    if MimiDecoder not in _original_methods:
        _original_methods[MimiDecoder] = MimiDecoder.forward
    MimiDecoder.forward = _patched_decoder_forward

    # Patch model methods
    if hasattr(model, '_decode_frame'):
        model._decode_frame = lambda *args, **kwargs: _patched_decode_frame(model, *args, **kwargs)

    if hasattr(model, 'decode'):
        model.decode = lambda *args, **kwargs: _patched_decode(model, *args, **kwargs)


def disable_streaming_decode() -> None:
    """Restore original methods (disable streaming patches)."""
    for cls, original in _original_methods.items():
        cls.forward = original
    _original_methods.clear()
