from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
from torch import nn
from transformers import MimiModel


DEFAULT_MIMI_MODEL_ID = "kyutai/mimi"


@dataclass(frozen=True)
class MimiConfig:
    model_id: str = DEFAULT_MIMI_MODEL_ID
    dtype: Optional[torch.dtype] = None


@dataclass
class StreamingDecoderState:
    """State for streaming audio decode.

    Contains both the transformer KV cache and the conv padding cache
    for proper stateful streaming.
    """
    # Transformer KV cache for attention continuity
    kv_cache: Optional[Any] = None
    # Conv padding cache for seamless audio boundaries
    padding_cache: Optional[Any] = None


class MimiCodec(nn.Module):
    """Thin wrapper around transformers' MimiModel for decoding audio tokens.

    Supports streaming decode with full state preservation:
    - Transformer KV cache (for attention continuity)
    - Conv padding cache (for seamless audio at chunk boundaries)

    The conv padding cache properly handles both Conv1d and ConvTranspose1d
    layers by caching input samples and trimming redundant output.
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
        import sys

        # Load directly to GPU with device_map to avoid CPU->GPU copy
        model = MimiModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": device},  # Load directly to target device
        )
        model.eval()
        # Enable use_cache for transformer KV cache
        model.config.use_cache = True

        streaming_enabled = False
        if enable_streaming:
            try:
                from .mimi_streaming import enable_streaming_decode
                enable_streaming_decode(model)
                streaming_enabled = True
                print("[MimiCodec] Streaming decode enabled with conv state caching", file=sys.stderr)
            except Exception as e:
                print(f"[MimiCodec] Warning: Could not enable streaming decode: {e}", file=sys.stderr)
                print("[MimiCodec] Falling back to KV-cache-only streaming", file=sys.stderr)

        return cls(model, device, streaming_enabled)

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
        """Decode audio codes to waveform with state preservation for streaming.

        This method maintains both transformer KV cache and conv padding cache
        across calls, enabling smooth transitions between audio chunks.

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)
            decoder_state: Previous StreamingDecoderState, or None for first chunk

        Returns:
            Tuple of (audio_waveform, new_decoder_state)
            - audio_waveform: Decoded audio tensor
            - new_decoder_state: State to pass to next decode_with_state() call
        """
        codes = codes.to(self.device)

        # Initialize state if needed
        if decoder_state is None:
            decoder_state = StreamingDecoderState()

        with torch.inference_mode():
            if self.streaming_enabled:
                # Create padding cache if needed
                if decoder_state.padding_cache is None:
                    from .mimi_streaming import MimiDecoderPaddingCache
                    decoder_state.padding_cache = MimiDecoderPaddingCache(self.model)

                # Decode with both KV cache and padding cache
                audio, new_kv_cache = self.model.decode(
                    codes,
                    decoder_past_key_values=decoder_state.kv_cache,
                    decoder_padding_cache=decoder_state.padding_cache,
                    return_dict=False,
                )

                new_state = StreamingDecoderState(
                    kv_cache=new_kv_cache,
                    padding_cache=decoder_state.padding_cache,  # Same cache object, updated in place
                )
            else:
                # Fallback: KV cache only
                audio, new_kv_cache = self.model.decode(
                    codes,
                    decoder_past_key_values=decoder_state.kv_cache,
                    return_dict=False,
                )
                new_state = StreamingDecoderState(kv_cache=new_kv_cache)

            return torch.clamp(audio, -1.0, 1.0), new_state

    def decode_with_lookahead(
        self,
        codes: torch.Tensor,
        output_frames: int,
        decoder_state: Optional[StreamingDecoderState] = None,
    ) -> Tuple[torch.Tensor, StreamingDecoderState]:
        """Decode audio codes with lookahead frames for forward context.

        This method enables overlap decoding to eliminate boundary artifacts:
        - Pass N frames of codes (e.g., 2 frames)
        - Get audio output for only the first `output_frames` frames (e.g., 1 frame)
        - The extra frames provide forward context to non-causal convolutions
        - State is updated based on output_frames only (not lookahead)

        Example usage for streaming with 1-frame lookahead:
            # Hold one frame, when next arrives, decode both but output first only
            codes_pair = torch.cat([held_frame, new_frame], dim=-1)  # 2 frames
            audio, state = codec.decode_with_lookahead(codes_pair, output_frames=1, decoder_state=state)
            held_frame = new_frame  # New frame becomes held frame for next iteration

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)
                   where seq_len > output_frames (extra frames are lookahead)
            output_frames: Number of frames to actually output audio for.
                          Must be less than codes.shape[-1].
            decoder_state: Previous StreamingDecoderState, or None for first chunk

        Returns:
            Tuple of (audio_waveform, new_decoder_state)
            - audio_waveform: Decoded audio for output_frames only
            - new_decoder_state: State to pass to next call (reflects output_frames processing)
        """
        codes = codes.to(self.device)
        num_input_frames = codes.shape[-1]

        if output_frames >= num_input_frames:
            # No lookahead, use regular decode
            return self.decode_with_state(codes, decoder_state)

        # Initialize state if needed
        if decoder_state is None:
            decoder_state = StreamingDecoderState()

        with torch.inference_mode():
            if self.streaming_enabled:
                # Create padding cache if needed
                if decoder_state.padding_cache is None:
                    from .mimi_streaming import MimiDecoderPaddingCache
                    decoder_state.padding_cache = MimiDecoderPaddingCache(self.model)

                # Decode with output_frames limit - lookahead provides forward context
                # but we only produce audio and update state for output_frames
                audio, new_kv_cache = self.model.decode(
                    codes,
                    decoder_past_key_values=decoder_state.kv_cache,
                    decoder_padding_cache=decoder_state.padding_cache,
                    output_frames=output_frames,
                    return_dict=False,
                )

                new_state = StreamingDecoderState(
                    kv_cache=new_kv_cache,
                    padding_cache=decoder_state.padding_cache,
                )
            else:
                # Fallback: no streaming support, decode and slice
                audio, new_kv_cache = self.model.decode(
                    codes,
                    decoder_past_key_values=decoder_state.kv_cache,
                    return_dict=False,
                )
                # Slice to output_frames worth of audio
                samples_to_keep = output_frames * self.samples_per_frame
                audio = audio[..., :samples_to_keep]
                new_state = StreamingDecoderState(kv_cache=new_kv_cache)

            return torch.clamp(audio, -1.0, 1.0), new_state

    def update_state(
        self,
        codes: torch.Tensor,
        decoder_state: Optional[StreamingDecoderState] = None,
    ) -> StreamingDecoderState:
        """Update decoder state without returning audio (for overlap decode).

        Runs the decoder forward pass to update KV cache and conv padding cache,
        but skips audio output processing. Use this when you need the state
        but will discard the audio anyway.

        Args:
            codes: Audio codes tensor of shape (batch, num_codebooks, seq_len)
            decoder_state: Previous StreamingDecoderState, or None for first chunk

        Returns:
            Updated StreamingDecoderState
        """
        codes = codes.to(self.device)

        if decoder_state is None:
            decoder_state = StreamingDecoderState()

        with torch.inference_mode():
            if self.streaming_enabled:
                if decoder_state.padding_cache is None:
                    from .mimi_streaming import MimiDecoderPaddingCache
                    decoder_state.padding_cache = MimiDecoderPaddingCache(self.model)

                # Run decode to update state, discard audio
                _, new_kv_cache = self.model.decode(
                    codes,
                    decoder_past_key_values=decoder_state.kv_cache,
                    decoder_padding_cache=decoder_state.padding_cache,
                    return_dict=False,
                )

                return StreamingDecoderState(
                    kv_cache=new_kv_cache,
                    padding_cache=decoder_state.padding_cache,
                )
            else:
                _, new_kv_cache = self.model.decode(
                    codes,
                    decoder_past_key_values=decoder_state.kv_cache,
                    return_dict=False,
                )
                return StreamingDecoderState(kv_cache=new_kv_cache)

    def encode(self, audio: torch.Tensor, *, return_dict: bool = False):
        audio = audio.to(self.device)
        with torch.inference_mode():
            return self.model.encode(audio, return_dict=return_dict)
