from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
from torch import nn
from transformers import MimiModel


DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"

# Type alias for decoder state (Cache object from transformers)
DecoderState = Any  # transformers.Cache, but we don't want to import it


@dataclass(frozen=True)
class MimiConfig:
    model_id: str = DEFAULT_MIMI_MODEL_ID
    dtype: Optional[torch.dtype] = None


class MimiCodec(nn.Module):
    """Thin wrapper around transformers' MimiModel for decoding audio tokens."""

    def __init__(self, model: MimiModel, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.device = device
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
    ) -> "MimiCodec":
        model = MimiModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
        model.eval()
        return cls(model, device)

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

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)
            decoder_state: Previous decoder state from last call, or None for first chunk

        Returns:
            Tuple of (audio_waveform, new_decoder_state)
            - audio_waveform: Decoded audio tensor
            - new_decoder_state: State to pass to next decode_with_state() call
        """
        codes = codes.to(self.device)
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
