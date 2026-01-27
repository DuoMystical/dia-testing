"""
Streaming support for Mimi decoder.

This module patches the HuggingFace transformers Mimi implementation to support
stateful streaming decoding. The decoder's convolutional layers need to maintain
state across chunks to avoid boundary artifacts.

The approach:
- For BOTH Conv1d and ConvTranspose1d: cache INPUT samples
- Prepend cached input when processing new chunk (provides context)
- For ConvTranspose1d: trim redundant output samples (they were already emitted)

This is simpler and more correct than trying to cache/blend outputs.

Usage:
    from dia2.audio.mimi_streaming import enable_streaming_decode, MimiDecoderPaddingCache

    # Enable streaming on a MimiModel instance
    enable_streaming_decode(model)

    # Create cache for streaming decode
    cache = MimiDecoderPaddingCache(model)

    # Decode with state
    audio1, kv = model.decode(codes1, decoder_padding_cache=cache, return_dict=False)
    audio2, kv = model.decode(codes2, decoder_past_key_values=kv, decoder_padding_cache=cache, return_dict=False)
"""

from __future__ import annotations

from typing import Optional, List, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ConvLayerCache:
    """Cache entry for a single convolutional layer.

    For both Conv1d and ConvTranspose1d, we cache INPUT samples.
    This provides the context needed for correct boundary handling.
    """
    # Number of input samples to cache (receptive field - 1)
    cache_size: int
    # For ConvTranspose1d: how many output samples to trim (cache_size * stride)
    output_trim: int = 0
    # Whether this is a transpose conv (affects output trimming)
    is_transpose: bool = False
    # The stride (for transpose conv output trim calculation)
    stride: int = 1
    # Cached input samples from previous chunk
    cached_input: Optional[torch.Tensor] = None
    # Track if this is the first chunk (no trimming needed)
    is_first_chunk: bool = True


class MimiDecoderPaddingCache:
    """
    Padding cache for streaming Mimi decoder.

    Manages input caching for all Conv1d and ConvTranspose1d layers
    to enable seamless chunk-by-chunk decoding.
    """

    def __init__(self, model: Any):
        """Initialize cache from a MimiModel instance."""
        self.layer_caches: List[ConvLayerCache] = []

        # Get decoder from model
        decoder = getattr(model, 'decoder', model)

        # Analyze decoder layers and create cache entries
        self._analyze_decoder(decoder)

    def _analyze_decoder(self, decoder: nn.Module):
        """Analyze decoder layers to determine caching requirements."""
        from transformers.models.mimi.modeling_mimi import (
            MimiConv1d,
            MimiConvTranspose1d,
            MimiResnetBlock,
        )

        layer_idx = 0

        def process_conv1d(layer: MimiConv1d) -> int:
            """Process a Conv1d layer, return its assigned index."""
            nonlocal layer_idx

            # For causal conv, we need padding_total input samples as context
            if layer.causal:
                cache_size = layer.padding_total
            else:
                cache_size = layer.padding_left

            self.layer_caches.append(ConvLayerCache(
                cache_size=cache_size,
                output_trim=0,  # Conv1d doesn't need output trimming
                is_transpose=False,
                stride=1,
            ))
            layer._streaming_layer_idx = layer_idx
            idx = layer_idx
            layer_idx += 1
            return idx

        def process_conv_transpose1d(layer: MimiConvTranspose1d) -> int:
            """Process a ConvTranspose1d layer, return its assigned index."""
            nonlocal layer_idx

            # Get the actual conv parameters
            conv = layer.conv
            kernel_size = conv.kernel_size[0]
            stride = conv.stride[0]

            # Cache size: number of input samples that affect boundary outputs
            # For transpose conv, kernel_size - 1 inputs affect the boundary
            cache_size = kernel_size - 1

            # Output trim: when we prepend cache_size inputs, we get
            # cache_size * stride extra output samples that were already emitted
            output_trim = cache_size * stride

            self.layer_caches.append(ConvLayerCache(
                cache_size=cache_size,
                output_trim=output_trim,
                is_transpose=True,
                stride=stride,
            ))
            layer._streaming_layer_idx = layer_idx
            idx = layer_idx
            layer_idx += 1
            return idx

        # Process all layers in the decoder
        for layer in decoder.layers:
            if isinstance(layer, MimiConv1d):
                process_conv1d(layer)
            elif isinstance(layer, MimiConvTranspose1d):
                process_conv_transpose1d(layer)
            elif isinstance(layer, MimiResnetBlock):
                # ResnetBlock contains Conv1d layers
                for sublayer in layer.block:
                    if isinstance(sublayer, MimiConv1d):
                        process_conv1d(sublayer)
                # Check shortcut too
                if layer.shortcut is not None and isinstance(layer.shortcut, MimiConv1d):
                    process_conv1d(layer.shortcut)

    def get_and_update(
        self,
        layer_idx: int,
        input_tensor: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], int]:
        """
        Get cached input to prepend, and update cache with current input.

        Args:
            layer_idx: Index of the layer
            input_tensor: Current input tensor (batch, channels, time)

        Returns:
            Tuple of (cached_input_to_prepend, output_samples_to_trim)
            - cached_input_to_prepend: Tensor to prepend, or None if first chunk
            - output_samples_to_trim: Number of output samples to trim (for transpose conv)
        """
        if layer_idx >= len(self.layer_caches):
            return None, 0

        cache = self.layer_caches[layer_idx]

        if cache.cache_size == 0:
            return None, 0

        # Get current cache to prepend (None for first chunk)
        cached_to_prepend = cache.cached_input

        # Calculate output trim (only for transpose conv, and not on first chunk)
        output_trim = 0
        if cache.is_transpose and not cache.is_first_chunk and cached_to_prepend is not None:
            output_trim = cache.output_trim

        # Update cache with the last cache_size samples from current input
        seq_len = input_tensor.shape[-1]
        if seq_len >= cache.cache_size:
            cache.cached_input = input_tensor[..., -cache.cache_size:].clone()
        else:
            # Input shorter than cache size - combine with existing cache
            if cache.cached_input is not None:
                combined = torch.cat([cache.cached_input, input_tensor], dim=-1)
                cache.cached_input = combined[..., -cache.cache_size:].clone()
            else:
                # First chunk and input is too short - just cache what we have
                cache.cached_input = input_tensor.clone()

        # Mark that we've processed at least one chunk
        cache.is_first_chunk = False

        return cached_to_prepend, output_trim

    def reset(self):
        """Reset all cached states for a new decode session."""
        for cache in self.layer_caches:
            cache.cached_input = None
            cache.is_first_chunk = True


def _patched_conv1d_forward(
    self,
    hidden_states: torch.Tensor,
    padding_cache: Optional[MimiDecoderPaddingCache] = None,
) -> torch.Tensor:
    """Patched forward for MimiConv1d with streaming support."""
    layer_idx = getattr(self, '_streaming_layer_idx', None)

    # Get extra padding needed for alignment
    extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

    if padding_cache is not None and layer_idx is not None and self.causal:
        # Get cached input and update cache
        cached_input, _ = padding_cache.get_and_update(layer_idx, hidden_states)

        if cached_input is not None:
            # Prepend cached input - this provides the context the conv needs
            hidden_states = torch.cat([cached_input, hidden_states], dim=-1)
            # Only add extra padding on the right if needed
            if extra_padding > 0:
                hidden_states = self._pad1d(hidden_states, (0, extra_padding), self.pad_mode)
        else:
            # First chunk - use normal padding
            hidden_states = self._pad1d(
                hidden_states, (self.padding_total, extra_padding), self.pad_mode
            )
    elif self.causal:
        # No cache, use normal causal padding
        hidden_states = self._pad1d(
            hidden_states, (self.padding_total, extra_padding), self.pad_mode
        )
    else:
        # Non-causal conv
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

    # Get cached input and output trim amount
    output_trim = 0
    if padding_cache is not None and layer_idx is not None:
        cached_input, output_trim = padding_cache.get_and_update(layer_idx, hidden_states)

        if cached_input is not None:
            # Prepend cached input for context
            hidden_states = torch.cat([cached_input, hidden_states], dim=-1)

    # Run the transpose convolution
    hidden_states = self.conv(hidden_states)

    # Standard end trimming
    end = hidden_states.shape[-1] - self.padding_right

    # Calculate start position: normal padding_left PLUS any streaming trim
    start = self.padding_left + output_trim

    # Ensure we don't trim more than we have
    if start >= end:
        # Edge case: would trim everything. This shouldn't happen in normal use.
        # Return at least 1 sample to avoid empty tensor issues
        start = max(0, end - 1)

    hidden_states = hidden_states[..., start:end]
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
        if isinstance(layer, (MimiConv1d, MimiConvTranspose1d)):
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
