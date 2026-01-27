from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
from torch import nn
from transformers import MimiModel


DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"

# Overlap frames for streaming decode (provides context for conv layers)
# This determines how many frames from the previous chunk we include when decoding
# 8 frames at 12.5Hz = 640ms of context, enough for most conv receptive fields
STREAMING_OVERLAP_FRAMES = 8


@dataclass(frozen=True)
class MimiConfig:
    model_id: str = DEFAULT_MIMI_MODEL_ID
    dtype: Optional[torch.dtype] = None


@dataclass
class StreamingDecoderState:
    """State for streaming audio decode with overlap-add."""
    # Last N frames of tokens from previous chunk (for context)
    overlap_tokens: Optional[torch.Tensor] = None
    # Number of audio samples already emitted (for trimming overlap)
    samples_emitted: int = 0


class MimiCodec(nn.Module):
    """Thin wrapper around transformers' MimiModel for decoding audio tokens.

    Supports streaming decode via overlap-add:
    - Prepend overlapping frames from previous chunk for context
    - Decode combined chunk
    - Trim overlapping samples from output

    This approach is simpler and more reliable than patching conv layer internals.
    """

    def __init__(self, model: MimiModel, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        cfg = getattr(model, "config", None)
        self.sample_rate = getattr(cfg, "sampling_rate", 24000)
        self.frame_rate = getattr(cfg, "frame_rate", 12.5)
        self.samples_per_frame = int(round(self.sample_rate / self.frame_rate)) if self.frame_rate else 0
        self.overlap_frames = STREAMING_OVERLAP_FRAMES

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MIMI_MODEL_ID,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        enable_streaming: bool = True,  # Kept for API compatibility, always uses overlap-add now
    ) -> "MimiCodec":
        import sys

        # Load directly to GPU with device_map to avoid CPU->GPU copy
        model = MimiModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device},  # Load directly to target device
        )
        model.eval()

        print(f"[MimiCodec] Loaded with overlap-add streaming ({STREAMING_OVERLAP_FRAMES} frame overlap)", file=sys.stderr)

        return cls(model, device)

    def decode(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """Decode audio codes to waveform (stateless, for non-streaming use).

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)

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
        decoder_state: Optional[StreamingDecoderState] = None,
    ) -> Tuple[torch.Tensor, StreamingDecoderState]:
        """Decode audio codes to waveform with overlap-add for seamless streaming.

        This method uses overlap-add to ensure smooth audio transitions:
        1. Prepend overlapping frames from previous chunk (provides conv context)
        2. Decode the combined tokens
        3. Trim the overlapping audio samples from output

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)
            decoder_state: Previous state from last call, or None for first chunk

        Returns:
            Tuple of (audio_waveform, new_decoder_state)
            - audio_waveform: Decoded audio tensor (only NEW samples, overlap trimmed)
            - new_decoder_state: State to pass to next decode_with_state() call
        """
        codes = codes.to(self.device)
        num_new_frames = codes.shape[-1]

        # Initialize state if needed
        if decoder_state is None:
            decoder_state = StreamingDecoderState()

        with torch.inference_mode():
            # Prepend overlap tokens from previous chunk (if available)
            if decoder_state.overlap_tokens is not None:
                # Combine: [overlap_tokens | new_tokens]
                combined_codes = torch.cat([decoder_state.overlap_tokens, codes], dim=-1)
                overlap_samples = decoder_state.overlap_tokens.shape[-1] * self.samples_per_frame
            else:
                combined_codes = codes
                overlap_samples = 0

            # Decode the combined chunk
            audio, _ = self.model.decode(combined_codes, return_dict=False)
            audio = torch.clamp(audio, -1.0, 1.0)

            # Trim the overlap samples from the start (already emitted in previous chunk)
            if overlap_samples > 0 and audio.shape[-1] > overlap_samples:
                audio = audio[..., overlap_samples:]

            # Save the last N frames as overlap for next chunk
            if num_new_frames >= self.overlap_frames:
                new_overlap = codes[..., -self.overlap_frames:]
            else:
                # Not enough new frames - combine with existing overlap
                if decoder_state.overlap_tokens is not None:
                    combined = torch.cat([decoder_state.overlap_tokens, codes], dim=-1)
                    new_overlap = combined[..., -self.overlap_frames:]
                else:
                    new_overlap = codes

            # Update state
            new_state = StreamingDecoderState(
                overlap_tokens=new_overlap,
                samples_emitted=decoder_state.samples_emitted + audio.shape[-1],
            )

            return audio, new_state

    def encode(self, audio: torch.Tensor, *, return_dict: bool = False):
        audio = audio.to(self.device)
        with torch.inference_mode():
            return self.model.encode(audio, return_dict=return_dict)
