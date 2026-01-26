from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
from torch import nn
from transformers import MimiModel


DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"

# Type alias for decoder state
# For streaming, this is a tuple of (kv_cache, padding_cache)
DecoderState = Any


@dataclass(frozen=True)
class MimiConfig:
    model_id: str = DEFAULT_MIMI_MODEL_ID
    dtype: Optional[torch.dtype] = None


class MimiCodec(nn.Module):
    """Thin wrapper around transformers' MimiModel for decoding audio tokens.

    Supports streaming decode with full state preservation including:
    - Transformer KV cache (for attention continuity)
    - Decoder convolutional padding cache (for seamless audio at chunk boundaries)
    """

    def __init__(self, model: MimiModel, device: torch.device, streaming_enabled: bool = False) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.streaming_enabled = streaming_enabled
        cfg = getattr(model, "config", None)
        self.sample_rate = getattr(cfg, "sampling_rate", 24000)
        self.frame_rate = getattr(cfg, "frame_rate", 12.5)
        self.samples_per_frame = int(round(self.sample_rate / self.frame_rate)) if self.frame_rate else 0

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MIMI_MODEL_ID,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        enable_streaming: bool = True,
    ) -> "MimiCodec":
        model = MimiModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()
        # Enable use_cache so decoder returns past_key_values for streaming
        model.config.use_cache = True

        streaming_enabled = False
        if enable_streaming:
            try:
                from .mimi_streaming import enable_streaming_decode
                enable_streaming_decode(model)
                streaming_enabled = True
                print("[MimiCodec] Streaming decode enabled with convolutional padding cache")
            except Exception as e:
                print(f"[MimiCodec] Warning: Could not enable streaming decode: {e}")
                print("[MimiCodec] Falling back to KV-cache-only streaming")

        return cls(model, device, streaming_enabled)

    def decode(
        self,
        codes: torch.Tensor,
        decoder_state: Optional[DecoderState] = None,
    ) -> torch.Tensor:
        """Decode audio codes to waveform (stateless, for backward compatibility).

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)
            decoder_state: Ignored in this method. Use decode_with_state() for streaming.

        Returns:
            Audio waveform tensor
        """
        codes = codes.to(self.device)
        with torch.inference_mode():
            audio, _ = self.model.decode(codes, return_dict=False)
            return torch.clamp(audio, -1.0, 1.0)

    def decode_with_state(
        self,
        codes: torch.Tensor,
        decoder_state: Optional[DecoderState] = None,
    ) -> Tuple[torch.Tensor, DecoderState]:
        """Decode audio codes to waveform with state preservation for streaming.

        This method maintains decoder state across calls, enabling smooth
        transitions between audio chunks without boundary artifacts.

        When streaming is enabled, the decoder state includes both:
        - Transformer KV cache (for attention continuity)
        - Convolutional padding cache (for seamless audio boundaries)

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)
            decoder_state: Previous decoder state from last call, or None for first chunk
                          Format: (kv_cache, padding_cache) if streaming enabled, else just kv_cache

        Returns:
            Tuple of (audio_waveform, new_decoder_state)
            - audio_waveform: Decoded audio tensor
            - new_decoder_state: State to pass to next decode_with_state() call
        """
        codes = codes.to(self.device)

        # Unpack state
        if self.streaming_enabled:
            if decoder_state is not None:
                kv_cache, padding_cache = decoder_state
            else:
                kv_cache = None
                # Create new padding cache for this decode session
                from .mimi_streaming import MimiDecoderPaddingCache
                padding_cache = MimiDecoderPaddingCache(self.model)

            with torch.inference_mode():
                audio, new_kv_cache = self.model.decode(
                    codes,
                    decoder_past_key_values=kv_cache,
                    decoder_padding_cache=padding_cache,
                    return_dict=False,
                )
                new_state = (new_kv_cache, padding_cache)
                return torch.clamp(audio, -1.0, 1.0), new_state
        else:
            # Fallback: KV cache only (original behavior)
            with torch.inference_mode():
                audio, new_state = self.model.decode(
                    codes,
                    decoder_past_key_values=decoder_state,
                    return_dict=False,
                )
                return torch.clamp(audio, -1.0, 1.0), new_state

    def encode(self, audio: torch.Tensor, *, return_dict: bool = False):
        audio = audio.to(self.device)
        with torch.inference_mode():
            return self.model.encode(audio, return_dict=return_dict)
