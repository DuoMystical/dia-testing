from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ..core.cache import KVCache
from ..core.model import DecodeState
from ..generation import (
    GenerationConfig,
    StreamingConfig,
    AudioChunkEvent,
    StatusEvent,
    CompleteEvent,
    ErrorEvent,
    StreamEvent,
    StreamGenerator,
)
from ..audio.grid import delay_frames, mask_audio_logits, undelay_frames
from .context import RuntimeContext
from .state_machine import State, TokenIds
from .guidance import apply_classifier_guidance, sample_audio_logits
from .sampler import sample_token
from .voice_clone import PrefixPlan
from .logger import RuntimeLogger

_GRAPH_CUBLAS_READY = False


class WebMOpusStreamer:
    """Streaming WebM/Opus encoder for continuous audio streaming.

    Creates a single continuous WebM stream with Opus audio that can be
    played via MediaSource Extensions in the browser.
    """

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        """Initialize the WebM streamer.

        Args:
            sample_rate: Audio sample rate (default 24000 for Dia)
            channels: Number of audio channels (default 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self._container = None
        self._stream = None
        self._output_buffer = None
        self._pts = 0  # Presentation timestamp
        self._initialized = False
        self._bytes_returned = 0  # Track how many bytes we've already returned

    def _ensure_initialized(self):
        """Initialize the encoder on first use."""
        if self._initialized:
            return

        import av
        import io

        # Create in-memory output buffer
        self._output_buffer = io.BytesIO()

        # Create WebM container with immediate flush settings
        # cluster_time_limit forces a new cluster after this many microseconds,
        # preventing the muxer from buffering multiple chunks worth of audio
        self._container = av.open(
            self._output_buffer,
            mode='w',
            format='webm',
            options={'cluster_time_limit': '1'}  # 1 microsecond = flush immediately
        )

        # Add Opus audio stream - layout determines channels (channels attr is read-only)
        self._stream = self._container.add_stream('libopus', rate=self.sample_rate)
        self._stream.layout = 'mono' if self.channels == 1 else 'stereo'

        self._initialized = True
        self._pts = 0
        self._bytes_returned = 0

    def get_init_segment(self) -> bytes:
        """Get the WebM initialization segment (header).

        Must be called before encoding any audio. Returns the WebM header
        that should be sent to the client first.
        """
        self._ensure_initialized()

        # Write header by flushing container state
        # The init segment is written when the first packet is muxed
        # For now, return empty - we'll include header with first audio
        return b""

    def encode_audio(self, audio_np) -> bytes:
        """Encode audio samples and return WebM segment data.

        Args:
            audio_np: Numpy array of float samples in [-1.0, 1.0]

        Returns:
            WebM segment bytes to append to the stream
        """
        import numpy as np
        import av
        import sys

        self._ensure_initialized()

        # Clip and convert to format expected by encoder
        audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)

        # Create audio frame
        frame = av.AudioFrame.from_ndarray(
            audio_np.reshape(1, -1),  # Shape: (channels, samples)
            format='flt',
            layout='mono' if self.channels == 1 else 'stereo'
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts

        # Debug: log input
        print(f"[WEBM DEBUG] encode_audio: input_samples={len(audio_np)}, pts={self._pts}, bytes_returned_before={self._bytes_returned}", file=sys.stderr)

        self._pts += len(audio_np)

        # Encode and mux
        packets_encoded = 0
        for packet in self._stream.encode(frame):
            packets_encoded += 1
            self._container.mux(packet)

        print(f"[WEBM DEBUG] encode_audio: packets_encoded={packets_encoded}", file=sys.stderr)

        # Read all bytes from buffer and return only the new ones
        self._output_buffer.seek(0)
        all_bytes = self._output_buffer.read()
        new_bytes = all_bytes[self._bytes_returned:]

        print(f"[WEBM DEBUG] encode_audio: buffer_total={len(all_bytes)}, new_bytes={len(new_bytes)}", file=sys.stderr)

        self._bytes_returned = len(all_bytes)

        return new_bytes

    def prime_encoder(self, audio_np) -> None:
        """Prime the encoder with audio samples, discarding the output.

        This warms up the Opus encoder's predictive coding state so that
        the first real audio chunk doesn't have a click/pop artifact.
        """
        import numpy as np
        import av
        import sys

        self._ensure_initialized()

        audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)

        frame = av.AudioFrame.from_ndarray(
            audio_np.reshape(1, -1),
            format='flt',
            layout='mono' if self.channels == 1 else 'stereo'
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        self._pts += len(audio_np)

        # Encode and mux
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

        # DON'T discard output - the WebM header is included in these bytes!
        # If we discard them, the browser's SourceBuffer will fail to decode.
        # The primed audio (~80ms of transition) will be included in first chunk,
        # but that's acceptable to ensure the stream plays correctly.
        self._output_buffer.seek(0)
        all_bytes = self._output_buffer.read()
        # Keep _bytes_returned at 0 so header is included in first encode_audio output

        print(f"[WEBM DEBUG] prime_encoder: primed with {len(audio_np)} samples, header+prime={len(all_bytes)} bytes (kept)", file=sys.stderr)

    def finalize(self) -> bytes:
        """Finalize the stream and return any remaining data."""
        import sys

        if not self._initialized:
            return b""

        # Flush encoder
        for packet in self._stream.encode(None):
            self._container.mux(packet)

        # Close container to finalize the WebM file
        self._container.close()

        # Get only the NEW bytes (after what we've already returned)
        self._output_buffer.seek(0)
        all_bytes = self._output_buffer.read()
        final_bytes = all_bytes[self._bytes_returned:]

        print(f"[WEBM DEBUG] finalize: buffer_total={len(all_bytes)}, new_bytes={len(final_bytes)}", file=sys.stderr)

        self._initialized = False
        self._container = None
        self._stream = None
        self._output_buffer = None
        self._bytes_returned = 0

        return final_bytes

    def reset(self):
        """Reset the streamer for a new stream."""
        if self._container is not None:
            try:
                self._container.close()
            except:
                pass
        self._initialized = False
        self._container = None
        self._stream = None
        self._output_buffer = None
        self._pts = 0
        self._bytes_returned = 0


# Global streamer instance (reset per generation session)
_webm_streamer: Optional[WebMOpusStreamer] = None


def get_webm_streamer(sample_rate: int = 24000) -> WebMOpusStreamer:
    """Get or create the global WebM streamer."""
    global _webm_streamer
    if _webm_streamer is None or _webm_streamer.sample_rate != sample_rate:
        _webm_streamer = WebMOpusStreamer(sample_rate=sample_rate)
    return _webm_streamer


def reset_webm_streamer():
    """Reset the WebM streamer for a new generation session."""
    global _webm_streamer
    if _webm_streamer is not None:
        _webm_streamer.reset()


def _ensure_graph_cublas_ready(device: torch.device) -> None:
    global _GRAPH_CUBLAS_READY
    if _GRAPH_CUBLAS_READY or device.type != "cuda":
        return
    tmp = torch.empty((1, 1), device=device, dtype=torch.float32)
    torch.matmul(tmp, tmp)
    torch.cuda.synchronize()
    _GRAPH_CUBLAS_READY = True
@dataclass
class GenerationState:
    decode: DecodeState
    step_tokens: torch.Tensor
    audio_buf: torch.Tensor

    def trim_audio(self, limit: int, pad_token: int, ungenerated: int) -> torch.Tensor:
        trimmed = self.audio_buf[:, :, :limit]
        pad = torch.full_like(trimmed, pad_token)
        trimmed = torch.where(trimmed == ungenerated, pad, trimmed)
        self.audio_buf = trimmed
        return trimmed

    @property
    def transformer_cache(self) -> KVCache:
        return self.decode.transformer

    @transformer_cache.setter
    def transformer_cache(self, cache: KVCache) -> None:
        self.decode.transformer = cache

    @property
    def depformer_cache(self) -> KVCache:
        return self.decode.depformer

    @depformer_cache.setter
    def depformer_cache(self, cache: KVCache) -> None:
        self.decode.depformer = cache

    def reset_dep_cache(self) -> None:
        self.decode.depformer.reset()

    def clone(self) -> "GenerationState":
        """Create a deep copy of this generation state for caching."""
        return GenerationState(
            decode=self.decode.clone(),
            step_tokens=self.step_tokens.clone(),
            audio_buf=self.audio_buf.clone(),
        )


@dataclass
class NetworkBuffers:
    text: torch.Tensor
    cb0: torch.Tensor
    dep: list[torch.Tensor]


def _allocate_network_buffers(runtime: RuntimeContext, branches: int) -> NetworkBuffers:
    device = runtime.device
    logits_dtype = runtime.precision.logits
    data_cfg = runtime.config.data
    text_logits = torch.empty((branches, 1, data_cfg.action_vocab_size), dtype=logits_dtype, device=device)
    cb0_logits = torch.empty((branches, 1, data_cfg.audio_vocab_size), dtype=logits_dtype, device=device)
    dep_vocab = runtime.model.depformer.audio_vocab_limit or data_cfg.audio_vocab_size
    dep_logits = [
        torch.empty((branches, 1, 1, dep_vocab), dtype=logits_dtype, device=device)
        for _ in range(runtime.model.depformer.num_depth)
    ]
    return NetworkBuffers(text=text_logits, cb0=cb0_logits, dep=dep_logits)


def build_initial_state(
    runtime: RuntimeContext,
    *,
    prefix: PrefixPlan | None = None,
) -> GenerationState:
    dep_q = runtime.model.depformer.num_audio_channels
    channels = 2 + dep_q
    branches = 2
    token_ids = runtime.constants
    step_tokens = torch.full(
        (branches, channels, 1),
        token_ids.pad,
        dtype=torch.long,
        device=runtime.device,
    )
    step_tokens[0, 0, 0] = token_ids.bos
    step_tokens[0, 1, 0] = token_ids.pad
    step_tokens[1, 0, 0] = token_ids.zero
    step_tokens[1, 1, 0] = token_ids.pad
    prefix_len = 0
    if prefix is not None:
        delayed = delay_frames(prefix.aligned_tokens, runtime.audio_delays, token_ids.audio_pad)
        prefix_len = delayed.shape[1]
    limit = runtime.config.runtime.max_context_steps
    total_steps = max(limit + prefix_len + 1, limit)
    decode_state = runtime.model.init_state(branches, runtime.device, total_steps)
    audio_buf = torch.full(
        (branches, dep_q, total_steps),
        token_ids.ungenerated,
        dtype=torch.long,
        device=runtime.device,
    )
    if prefix is not None:
        delayed = delay_frames(prefix.aligned_tokens, runtime.audio_delays, token_ids.audio_pad).to(runtime.device)
        audio_buf[0, :, : delayed.shape[1]] = delayed
        if branches > 1:
            audio_buf[1:, :, : delayed.shape[1]] = delayed
    return GenerationState(decode_state, step_tokens, audio_buf)


def _fill_audio_channels(
    step_tokens: torch.Tensor,
    audio_buf: torch.Tensor,
    delays: torch.Tensor,
    step: int,
    bos_token: int,
) -> None:
    channels = delays.numel()
    if channels == 0:
        return
    target = step_tokens[:, 2 : 2 + channels, 0]
    if step < audio_buf.shape[-1]:
        target.copy_(audio_buf[:, :channels, step])
    else:
        target.fill_(bos_token)
    mask = delays > step
    mask_expanded = mask.unsqueeze(0).expand_as(target)
    target.copy_(torch.where(mask_expanded, bos_token, target))


def _execute_transformer_step(
    step_tokens: torch.Tensor,
    positions_view: torch.Tensor,
    generation: GenerationState,
    transformer_step,
    buffers: NetworkBuffers,
) -> torch.Tensor:
    hidden_t, text_logits_t, cb0_logits_t, present = transformer_step(
        step_tokens,
        positions_view,
        generation.transformer_cache,
    )
    buffers.text.copy_(text_logits_t)
    buffers.cb0.copy_(cb0_logits_t)
    generation.transformer_cache = present
    return hidden_t


def _execute_depformer_stage(
    stage_index: int,
    prev_audio: torch.Tensor,
    hidden_t: torch.Tensor,
    generation: GenerationState,
    depformer_step,
    main_tokens: Optional[torch.Tensor],
    second_tokens: Optional[torch.Tensor],
    buffers: NetworkBuffers,
) -> None:
    logits_stage, dep_present = depformer_step(
        prev_audio=prev_audio,
        transformer_out=hidden_t,
        stage_index=stage_index,
        cache=generation.depformer_cache,
        main_text=main_tokens if stage_index == 0 else None,
        second_text=second_tokens if stage_index == 0 else None,
    )
    target = buffers.dep[stage_index]
    if logits_stage.shape != target.shape:
        raise RuntimeError(
            f"depformer logits shape mismatch: {logits_stage.shape} vs {target.shape}"
        )
    target.copy_(logits_stage)
    generation.depformer_cache = dep_present


def _execute_transformer_graph(
    runtime: RuntimeContext,
    step_tokens: torch.Tensor,
    positions_view: torch.Tensor,
    branches: int,
    generation: GenerationState,
    transformer_step,
    buffers: NetworkBuffers,
    transformer_capture: Optional[Tuple[torch.cuda.CUDAGraph, torch.Tensor]],
    dep_captures: Optional[list[dict]],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    if transformer_capture is None:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hidden_ref = _execute_transformer_step(
                step_tokens,
                positions_view,
                generation,
                transformer_step,
                buffers,
            )
        transformer_capture = (graph, hidden_ref)
        if runtime.model.depformer.num_depth > 0:
            dep_captures = []
            for idx in range(runtime.model.depformer.num_depth):
                capture = {
                    "graph": torch.cuda.CUDAGraph(),
                    "captured": False,
                    "prev_audio": torch.empty((branches,), dtype=torch.long, device=runtime.device),
                    "main_tokens": torch.empty((branches,), dtype=torch.long, device=runtime.device) if idx == 0 else None,
                    "second_tokens": torch.empty((branches,), dtype=torch.long, device=runtime.device) if idx == 0 else None,
                }
                dep_captures.append(capture)
    else:
        transformer_capture[0].replay()
    return transformer_capture, dep_captures

def _execute_depformer_graph(
    stage: int,
    prev_audio: torch.Tensor,
    hidden_t: torch.Tensor,
    generation: GenerationState,
    depformer_step,
    main_tokens: Optional[torch.Tensor],
    aux_tokens: Optional[torch.Tensor],
    buffers: NetworkBuffers,
    capture: dict[str, torch.Tensor | torch.cuda.CUDAGraph | bool],
) -> dict[str, torch.Tensor | torch.cuda.CUDAGraph | bool]:
    capture["prev_audio"].copy_(prev_audio)
    if capture["main_tokens"] is not None and stage == 0:
        capture["main_tokens"].copy_(main_tokens)
        capture["second_tokens"].copy_(aux_tokens)
    if not capture["captured"]:
        torch.cuda.synchronize()
        with torch.cuda.graph(capture["graph"]):
            _execute_depformer_stage(
                stage_index=stage,
                prev_audio=capture["prev_audio"],
                hidden_t=hidden_t,
                generation=generation,
                depformer_step=depformer_step,
                main_tokens=capture["main_tokens"],
                second_tokens=capture["second_tokens"],
                buffers=buffers,
            )
        capture["captured"] = True
    else:
        capture["graph"].replay()

    return capture


def run_generation_loop(
    runtime: RuntimeContext,
    *,
    state: State,
    generation: GenerationState,
    config: GenerationConfig,
    start_step: int = 0,
    logger: RuntimeLogger | None = None,
) -> tuple[Optional[int], torch.Tensor]:
    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps
    if max_context <= 0:
        raise ValueError("Runtime configuration must specify a positive max_context_steps")
    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    cfg_active = config.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    flush_tail = max_delay + getattr(runtime.machine, "max_padding", 0)
    first_word_frame: Optional[int] = None
    eos_cutoff: Optional[int] = None
    last_step = start_step - 1
    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    use_torch_compile = config.use_torch_compile and runtime.device.type == "cuda"
    transformer_needs_compiling = use_torch_compile
    depformer_needs_compiling = [use_torch_compile] * runtime.model.depformer.num_depth
    if use_torch_compile:
        sample_token_fn = torch.compile(sample_token, dynamic=True, mode="max-autotune", fullgraph=True)
        sample_audio_logits_fn = torch.compile(sample_audio_logits, dynamic=True, mode="max-autotune", fullgraph=True)
    else:
        sample_token_fn = sample_token
        sample_audio_logits_fn = sample_audio_logits
    transformer_step = runtime.transformer_step
    depformer_step = runtime.depformer_step
    buffers = _allocate_network_buffers(runtime, branches)
    positions_view = positions.expand(branches, -1)
    transformer_capture = None
    dep_captures: list[dict] | None = None
    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)
    processed_steps = 0
    report_interval = 12
    with torch.inference_mode():
        for offset in range(max_context):
            if use_torch_compile:
                torch.compiler.cudagraph_mark_step_begin()
            t = start_step + offset
            if eos_cutoff is not None and t >= eos_cutoff:
                break
            if t + 1 >= audio_buf.shape[-1]:
                break
            generation.reset_dep_cache()
            positions.fill_(t)
            _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)
            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad
            if transformer_needs_compiling or not use_graph:
                if transformer_needs_compiling:
                    # Must use -no-cudagraphs variant as we are manually using graphs too.
                    transformer_step = torch.compile(
                        runtime.transformer_step,
                        dynamic=True,
                        mode="max-autotune-no-cudagraphs",
                    )
                    transformer_needs_compiling = False
                hidden_t = _execute_transformer_step(
                    step_tokens,
                    positions_view,
                    generation,
                    transformer_step,
                    buffers,
                )
            else:
                transformer_capture, dep_captures = _execute_transformer_graph(
                    runtime=runtime,
                    step_tokens=step_tokens,
                    positions_view=positions_view,
                    branches=branches,
                    generation=generation,
                    transformer_step=transformer_step,
                    buffers=buffers,
                    transformer_capture=transformer_capture,
                    dep_captures=dep_captures,
                )
                hidden_t = transformer_capture[1]

            guided_text = apply_classifier_guidance(buffers.text, cfg_active, config.cfg_scale, config.cfg_filter_k)
            if guided_text.shape[0] > 1:
                guided_text = guided_text[:1]

            text_token = sample_token_fn(
                guided_text,
                temp=config.text.temperature,
                top_k=config.text.top_k,
            ).item()

            main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
            second_token = aux_token if aux_token != -1 else token_ids.pad
            if first_word_frame is None and main_token == token_ids.new_word:
                first_word_frame = t - config.initial_padding
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = second_token

            guided_cb0 = apply_classifier_guidance(buffers.cb0, cfg_active, config.cfg_scale, config.cfg_filter_k)
            if guided_cb0.shape[0] > 1:
                guided_cb0 = guided_cb0[:1]
            masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
            codebook_token = sample_audio_logits_fn(masked_cb0, config.audio.temperature, config.audio.top_k)
            audio_buf[:, 0, t + 1] = codebook_token

            prev_audio = codebook_token.expand(branches)
            main_tokens.fill_(main_token)
            aux_tokens.fill_(second_token)
            for stage in range(runtime.model.depformer.num_depth):
                if use_graph and dep_captures is not None:
                    if depformer_needs_compiling[stage]:
                        runtime.model.depformer._forward_stage = torch.compile(
                            runtime.model.depformer._forward_stage,
                            dynamic=True,
                            mode="max-autotune-no-cudagraphs",
                        )
                        depformer_needs_compiling[stage] = False
                        _execute_depformer_stage(
                            stage_index=stage,
                            prev_audio=prev_audio,
                            hidden_t=hidden_t,
                            generation=generation,
                            depformer_step=depformer_step,
                            main_tokens=main_tokens,
                            second_tokens=aux_tokens,
                            buffers=buffers,
                        )
                    else:
                        dep_captures[stage] = _execute_depformer_graph(
                            stage=stage,
                            prev_audio=prev_audio,
                            hidden_t=hidden_t,
                            generation=generation,
                            depformer_step=depformer_step,
                            main_tokens=main_tokens,
                            aux_tokens=aux_tokens,
                            buffers=buffers,
                            capture=dep_captures[stage],
                        )

                else:
                    _execute_depformer_stage(
                        stage_index=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=generation,
                        depformer_step=depformer_step,
                        main_tokens=main_tokens,
                        second_tokens=aux_tokens,
                        buffers=buffers,
                    )
                dep_logits = apply_classifier_guidance(buffers.dep[stage], cfg_active, config.cfg_scale, config.cfg_filter_k)
                if dep_logits.shape[0] > 1:
                    dep_logits = dep_logits[:1]
                stage_token = sample_audio_logits_fn(
                    dep_logits,
                    config.audio.temperature,
                    config.audio.top_k,
                )
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)
            last_step = t
            if eos_cutoff is None and state.end_step is not None:
                eos_cutoff = state.end_step + flush_tail
            processed_steps = offset + 1
            if logger and processed_steps % report_interval == 0:
                logger.progress(processed_steps, max_context)

    if logger and processed_steps and processed_steps % report_interval != 0:
        logger.progress(processed_steps, max_context)

    if first_word_frame is None:
        first_word_frame = start_step
    if last_step < start_step:
        limit = min(start_step + 1, audio_buf.shape[-1])
    else:
        limit = min(last_step + 2, audio_buf.shape[-1])
    trimmed = generation.trim_audio(limit, token_ids.audio_pad, token_ids.ungenerated)
    return first_word_frame, trimmed


def decode_audio(runtime: RuntimeContext, tokens: torch.Tensor) -> torch.Tensor:
    """Decode audio tokens to waveform (stateless)."""
    if tokens.shape[-1] == 0:
        return torch.zeros(0, device=runtime.device)
    with torch.inference_mode():
        pcm = runtime.mimi.decode(tokens.to(runtime.device))
        return pcm[0, 0]


def decode_audio_streaming(
    runtime: RuntimeContext,
    tokens: torch.Tensor,
    decoder_state=None,
):
    """Decode audio tokens to waveform with streaming state.

    Uses the MimiCodec's streaming decode to maintain continuity
    between chunks without boundary artifacts.

    Args:
        runtime: Runtime context containing the mimi codec
        tokens: Audio tokens to decode (batch, codebooks, frames)
        decoder_state: Previous decoder state, or None for first chunk

    Returns:
        Tuple of (waveform, new_decoder_state)
        - waveform: 1D tensor of audio samples
        - new_decoder_state: State to pass to next call
    """
    if tokens.shape[-1] == 0:
        return torch.zeros(0, device=runtime.device), decoder_state

    with torch.inference_mode():
        pcm, new_state = runtime.mimi.decode_with_state(
            tokens.to(runtime.device),
            decoder_state=decoder_state,
        )
        return pcm[0, 0], new_state


# Module-level tracking for cross-chunk boundary analysis
_last_chunk_end_samples = None
_chunk_diagnostics_enabled = False  # Set True for detailed per-chunk audio analysis
_chunk_counter = 0

# Debug mode for saving raw audio chunks
_debug_save_chunks = False
_debug_output_dir = None
_debug_all_samples = None  # Accumulate all samples for concatenated file


def reset_chunk_diagnostics():
    """Reset chunk diagnostics for a new generation session."""
    global _last_chunk_end_samples, _chunk_counter, _debug_all_samples
    _last_chunk_end_samples = None
    _chunk_counter = 0
    _debug_all_samples = []


def enable_debug_chunk_save(output_dir: str):
    """Enable saving raw audio chunks to files for offline analysis.

    Each chunk will be saved as chunk_NNNN.wav in the output directory.
    A concatenated file (all_chunks.wav) will also be saved at the end.

    Args:
        output_dir: Directory to save chunk files (will be created if needed)
    """
    import os
    import sys
    global _debug_save_chunks, _debug_output_dir, _debug_all_samples

    _debug_save_chunks = True
    _debug_output_dir = output_dir
    _debug_all_samples = []

    os.makedirs(output_dir, exist_ok=True)
    print(f"[DEBUG] Audio chunk saving enabled. Output dir: {output_dir}", file=sys.stderr)


def disable_debug_chunk_save():
    """Disable saving raw audio chunks."""
    global _debug_save_chunks, _debug_output_dir
    _debug_save_chunks = False
    _debug_output_dir = None


def save_debug_concatenated():
    """Save all accumulated samples as a single concatenated WAV file."""
    import sys
    import os
    import io
    import wave
    import numpy as np
    global _debug_all_samples, _debug_output_dir

    if not _debug_save_chunks or _debug_output_dir is None or not _debug_all_samples:
        return

    all_samples = np.concatenate(_debug_all_samples)
    all_samples = np.clip(all_samples, -1.0, 1.0)
    pcm16 = (all_samples * 32767.0).astype(np.int16)

    output_path = os.path.join(_debug_output_dir, "all_chunks_concatenated.wav")
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)  # Assuming 24kHz
        wav_file.writeframes(pcm16.tobytes())

    print(f"[DEBUG] Saved concatenated audio: {output_path} ({len(all_samples)} samples, {len(all_samples)/24000:.2f}s)", file=sys.stderr)


def _analyze_audio_chunk(audio_np, chunk_index: int, sample_rate: int):
    """Analyze an audio chunk for anomalies that could cause pops/clicks.

    Checks for:
    - Spikes at start/end (sudden amplitude changes)
    - DC offset
    - Discontinuities at chunk boundaries
    - Abnormal sample values
    """
    import sys
    import os
    import io
    import wave
    import numpy as np
    global _last_chunk_end_samples, _debug_save_chunks, _debug_output_dir, _debug_all_samples

    num_samples = len(audio_np)
    if num_samples == 0:
        return

    # Compute basic stats
    max_abs = np.max(np.abs(audio_np))
    mean_val = np.mean(audio_np)
    rms = np.sqrt(np.mean(audio_np ** 2))

    # Analyze start/end of chunk (first/last 10 samples)
    boundary_size = min(10, num_samples // 2)
    start_samples = audio_np[:boundary_size]
    end_samples = audio_np[-boundary_size:]

    start_max = np.max(np.abs(start_samples))
    end_max = np.max(np.abs(end_samples))

    # Calculate sample-to-sample differences to detect sudden jumps
    if num_samples > 1:
        diffs = np.abs(np.diff(audio_np))
        max_diff = np.max(diffs)
        # Find where the big jumps are (if any)
        jump_threshold = 0.1  # 10% of full scale
        big_jumps = np.where(diffs > jump_threshold)[0]
    else:
        max_diff = 0
        big_jumps = []

    # Check for discontinuity at chunk boundary (vs previous chunk)
    boundary_discontinuity = None
    if _last_chunk_end_samples is not None and len(_last_chunk_end_samples) > 0:
        prev_last = _last_chunk_end_samples[-1]
        curr_first = audio_np[0]
        boundary_discontinuity = abs(curr_first - prev_last)

    # Save end samples for next chunk comparison
    _last_chunk_end_samples = end_samples.copy()

    # Log diagnostics
    print(f"[AUDIO_DIAG] Chunk {chunk_index}: {num_samples} samples ({num_samples/sample_rate*1000:.1f}ms)", file=sys.stderr)
    print(f"  Stats: max_abs={max_abs:.4f}, mean={mean_val:.6f}, rms={rms:.4f}", file=sys.stderr)
    print(f"  Boundaries: start_max={start_max:.4f}, end_max={end_max:.4f}", file=sys.stderr)
    print(f"  First 5 samples: {audio_np[:5].tolist()}", file=sys.stderr)
    print(f"  Last 5 samples: {audio_np[-5:].tolist()}", file=sys.stderr)

    if boundary_discontinuity is not None:
        # Flag if discontinuity is large (potential click source)
        flag = " *** POSSIBLE CLICK ***" if boundary_discontinuity > 0.05 else ""
        print(f"  Boundary jump from prev chunk: {boundary_discontinuity:.4f}{flag}", file=sys.stderr)

    print(f"  Max sample-to-sample diff: {max_diff:.4f}", file=sys.stderr)

    if len(big_jumps) > 0:
        # Show where the big jumps occur
        jump_positions = big_jumps[:5]  # First 5 big jumps
        print(f"  Big jumps (>{jump_threshold}) at samples: {jump_positions.tolist()}", file=sys.stderr)
        for pos in jump_positions[:3]:
            if pos > 0 and pos < num_samples - 1:
                print(f"    Sample {pos}: {audio_np[pos-1]:.4f} -> {audio_np[pos]:.4f} -> {audio_np[pos+1]:.4f} (diff={diffs[pos]:.4f})", file=sys.stderr)

    # Save chunk to file if debug mode is enabled
    if _debug_save_chunks and _debug_output_dir is not None:
        # Save individual chunk
        chunk_path = os.path.join(_debug_output_dir, f"chunk_{chunk_index:04d}.wav")
        chunk_clipped = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (chunk_clipped * 32767.0).astype(np.int16)

        with wave.open(chunk_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

        print(f"  [DEBUG] Saved: {chunk_path}", file=sys.stderr)

        # Accumulate for concatenated file
        if _debug_all_samples is not None:
            _debug_all_samples.append(audio_np.copy())


def _encode_webm_chunk(
    waveform: torch.Tensor,
    sample_rate: int,
) -> bytes:
    """Encode a waveform tensor to WebM/Opus segment bytes.

    Uses a persistent WebM streamer to maintain encoder state across chunks,
    creating a single continuous stream that eliminates clicks/pops at chunk boundaries.
    """
    import numpy as np
    import sys
    global _chunk_counter

    audio_np = waveform.detach().cpu().numpy()

    # Run diagnostics before any processing
    if _chunk_diagnostics_enabled:
        _analyze_audio_chunk(audio_np, _chunk_counter, sample_rate)
        _chunk_counter += 1

    # Clip to valid range
    audio_np = np.clip(audio_np, -1.0, 1.0)

    # Ensure it's the right shape (1D array)
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()

    # Use persistent WebM streamer for seamless streaming
    streamer = get_webm_streamer(sample_rate)
    webm_bytes = streamer.encode_audio(audio_np)

    if _chunk_diagnostics_enabled:
        print(f"  [WEBM] samples={len(audio_np)}, sample_rate={sample_rate}Hz, size={len(webm_bytes)} bytes", file=sys.stderr)
        print(f"  [WEBM] first 5 samples: {audio_np[:5].tolist()}", file=sys.stderr)
        print(f"  [WEBM] last 5 samples: {audio_np[-5:].tolist()}", file=sys.stderr)

    return webm_bytes


def run_seed_warmup(
    runtime: RuntimeContext,
    generation: GenerationState,
    config: GenerationConfig,
    min_steps: int,
    warmup_state: Optional[State] = None,
    max_steps: int = 500,
) -> Tuple[GenerationState, int]:
    """
    Run warmup steps to establish seed-dependent state.

    This runs generation with a warmup phrase until all entries are consumed,
    building up the KV cache and audio buffer. The warmup phrase ensures
    consistent behavior across different user texts when using the same seed.

    Args:
        runtime: Runtime context
        generation: Initial generation state
        config: Generation configuration
        min_steps: Minimum warmup steps (should be >= max_delay for codec)
        warmup_state: State with warmup phrase entries (created by caller)
        max_steps: Maximum warmup steps (safety limit)

    Returns:
        Tuple of (generation_state, warmup_steps) where warmup_steps is the
        number of steps actually run.
    """
    import sys

    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]

    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    cfg_active = config.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor

    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    sample_token_fn = sample_token
    sample_audio_logits_fn = sample_audio_logits

    transformer_step = runtime.transformer_step
    depformer_step = runtime.depformer_step
    buffers = _allocate_network_buffers(runtime, branches)
    positions_view = positions.expand(branches, -1)

    transformer_capture = None
    dep_captures = None

    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)

    print(f"[WARMUP] Running warmup (min_steps={min_steps}, max_steps={max_steps})", file=sys.stderr)

    warmup_steps = 0
    with torch.inference_mode():
        for t in range(max_steps):
            generation.reset_dep_cache()
            positions.fill_(t)
            _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)

            if branches > 1:
                step_tokens[1:, 0, 0] = token_ids.zero
                step_tokens[1:, 1, 0] = token_ids.pad

            # Run transformer
            if use_graph and transformer_capture is not None:
                transformer_capture[0].replay()
                hidden_t = transformer_capture[1]
            else:
                if use_graph and transformer_capture is None:
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        hidden_t = _execute_transformer_step(
                            step_tokens, positions_view, generation,
                            transformer_step, buffers,
                        )
                    transformer_capture = (graph, hidden_t)
                    # Initialize dep_captures
                    dep_captures = []
                    for idx in range(runtime.model.depformer.num_depth):
                        dep_captures.append({
                            "graph": torch.cuda.CUDAGraph(),
                            "captured": False,
                            "prev_audio": torch.empty((branches,), dtype=torch.long, device=runtime.device),
                            "main_tokens": torch.empty((branches,), dtype=torch.long, device=runtime.device) if idx == 0 else None,
                            "second_tokens": torch.empty((branches,), dtype=torch.long, device=runtime.device) if idx == 0 else None,
                        })
                else:
                    hidden_t = _execute_transformer_step(
                        step_tokens, positions_view, generation,
                        transformer_step, buffers,
                    )

            # Sample text token
            guided_text = apply_classifier_guidance(buffers.text, cfg_active, config.cfg_scale, config.cfg_filter_k)
            if guided_text.shape[0] > 1:
                guided_text = guided_text[:1]
            text_token = sample_token_fn(guided_text, temp=config.text.temperature, top_k=config.text.top_k).item()

            # Process through state machine with warmup phrase (or use pad if no state)
            if warmup_state is not None:
                main_token, aux_token, _ = runtime.machine.process(t, warmup_state, text_token)
                second_token = aux_token if aux_token != -1 else token_ids.pad
            else:
                # Fallback: use pad tokens (old behavior)
                main_token = token_ids.pad
                second_token = token_ids.pad
            step_tokens[:, 0, 0] = main_token
            step_tokens[:, 1, 0] = second_token

            # Sample audio token (CB0)
            guided_cb0 = apply_classifier_guidance(buffers.cb0, cfg_active, config.cfg_scale, config.cfg_filter_k)
            if guided_cb0.shape[0] > 1:
                guided_cb0 = guided_cb0[:1]
            masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
            codebook_token = sample_audio_logits_fn(masked_cb0, config.audio.temperature, config.audio.top_k)
            audio_buf[:, 0, t + 1] = codebook_token

            # Run depformer stages
            prev_audio = codebook_token.expand(branches)
            main_tokens.fill_(main_token)
            aux_tokens.fill_(second_token)

            for stage in range(runtime.model.depformer.num_depth):
                if use_graph and dep_captures is not None:
                    dep_captures[stage] = _execute_depformer_graph(
                        stage=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=generation,
                        depformer_step=depformer_step,
                        main_tokens=main_tokens,
                        aux_tokens=aux_tokens,
                        buffers=buffers,
                        capture=dep_captures[stage],
                    )
                else:
                    _execute_depformer_stage(
                        stage_index=stage,
                        prev_audio=prev_audio,
                        hidden_t=hidden_t,
                        generation=generation,
                        depformer_step=depformer_step,
                        main_tokens=main_tokens,
                        second_tokens=aux_tokens,
                        buffers=buffers,
                    )

                dep_logits = apply_classifier_guidance(
                    buffers.dep[stage], cfg_active, config.cfg_scale, config.cfg_filter_k
                )
                if dep_logits.shape[0] > 1:
                    dep_logits = dep_logits[:1]
                stage_token = sample_audio_logits_fn(
                    dep_logits, config.audio.temperature, config.audio.top_k
                )
                audio_buf[:, stage + 1, t + 1] = stage_token
                prev_audio = stage_token.expand(branches)

            warmup_steps = t + 1

            # Check if warmup is complete: entries consumed AND minimum steps reached
            # Streaming loop handles flush_tail for output alignment separately
            if warmup_state is not None and warmup_state.end_step is not None:
                if warmup_steps >= min_steps:
                    print(f"[WARMUP] Entries consumed at step {warmup_state.end_step}, stopping at {warmup_steps} steps", file=sys.stderr)
                    break

    print(f"[WARMUP] Completed {warmup_steps} warmup steps", file=sys.stderr)
    return generation, warmup_steps


def run_streaming_generation_loop(
    runtime: RuntimeContext,
    *,
    state: State,
    generation: GenerationState,
    config: GenerationConfig,
    streaming_config: StreamingConfig,
    start_step: int = 0,
    logger: RuntimeLogger | None = None,
    decoder_state=None,
) -> StreamGenerator:
    """
    Streaming generation loop that yields audio chunks during generation.

    This function yields StreamEvent objects as the generation progresses,
    allowing real-time audio streaming.

    Args:
        runtime: Runtime context
        state: State machine state
        generation: Generation state
        config: Generation configuration
        streaming_config: Streaming configuration
        start_step: Starting step (after prefix warmup)
        logger: Optional logger
        decoder_state: Optional pre-initialized decoder state (for warmup cache)

    Yields:
        StreamEvent objects (AudioChunkEvent, StatusEvent, CompleteEvent, ErrorEvent)
    """
    import time
    import sys

    step_tokens = generation.step_tokens
    audio_buf = generation.audio_buf
    branches = step_tokens.shape[0]
    max_context = runtime.config.runtime.max_context_steps

    if max_context <= 0:
        yield ErrorEvent(error="Runtime configuration must specify a positive max_context_steps")
        return

    positions = torch.empty(1, 1, dtype=torch.long, device=runtime.device)
    main_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    aux_tokens = torch.empty(branches, dtype=torch.long, device=runtime.device)
    cfg_active = config.cfg_scale != 1.0
    token_ids = runtime.constants
    delay_tensor = runtime.audio_delay_tensor
    max_delay = int(delay_tensor.max().item()) if delay_tensor.numel() else 0
    # FIX: flush_tail only needs max_delay to flush audio through codec delay pipeline
    # Previously included max_padding which caused extra "hallucinated" audio at end
    flush_tail = max_delay
    first_word_frame: Optional[int] = None
    first_word_frame_applied = False  # Track if we've applied the skip for initial frames
    eos_cutoff: Optional[int] = None
    last_step = start_step - 1

    # Enable CUDA graph for streaming - it works with periodic audio decoding
    use_graph = config.use_cuda_graph and runtime.device.type == "cuda"
    use_torch_compile = config.use_torch_compile and runtime.device.type == "cuda"
    transformer_needs_compiling = use_torch_compile
    depformer_needs_compiling = [use_torch_compile] * runtime.model.depformer.num_depth

    if use_torch_compile:
        sample_token_fn = torch.compile(sample_token, dynamic=True, mode="max-autotune", fullgraph=True)
        sample_audio_logits_fn = torch.compile(sample_audio_logits, dynamic=True, mode="max-autotune", fullgraph=True)
    else:
        sample_token_fn = sample_token
        sample_audio_logits_fn = sample_audio_logits

    transformer_step = runtime.transformer_step
    depformer_step = runtime.depformer_step
    buffers = _allocate_network_buffers(runtime, branches)
    positions_view = positions.expand(branches, -1)

    # CUDA graph state for streaming
    transformer_capture = None
    dep_captures: list[dict] | None = None
    if use_graph:
        _ensure_graph_cublas_ready(runtime.device)

    # Check if we're on CUDA for sync operations
    is_cuda = runtime.device.type == "cuda"

    # Create separate CUDA stream for audio decoding (overlaps with generation)
    decode_stream = torch.cuda.Stream() if is_cuda else None

    # Pending async decode state: (waveform_tensor, chunk_index, timestamp_ms)
    pending_decode = None

    # Streaming state
    chunk_size = streaming_config.chunk_size_frames
    min_chunk = streaming_config.min_chunk_frames
    chunk_index = 0
    total_audio_ms = 0.0
    generation_start_time = time.time()

    # Decoder state for maintaining continuity between audio chunks
    # Use provided state (from warmup cache) or start fresh
    #
    # We track TWO separate positions:
    # 1. last_aligned_decoded - first aligned frame index to decode (decoder's next frame)
    # 2. last_aligned_emitted - last aligned frame we've outputted
    #
    # With warmup properly flushed, start_step = end_step + max_delay.
    # The decoder was primed during warmup on aligned frames 0 to (start_step - max_delay).
    #
    # Example with end_step=120, max_delay=18:
    # - warmup_steps = 120 + 18 = 138 (flushed for max_delay after text consumed)
    # - Decoder primed on aligned frames 0 to 120 (121 total)
    # - last_aligned_decoded = (138+1) - 18 = 121 (first frame to decode)
    # - Frame 121 uses tokens 121-139, position 139 is first user token
    # - last_aligned_emitted = 122 skips the transition frame (mostly warmup)
    # - Frame 122 uses tokens 122-140, has 2 user tokens (139, 140)
    #
    # First user audio starts at step start_step, writes to position start_step+1.
    if streaming_config.debug_include_warmup:
        # Debug mode: output ALL audio including warmup, decode from frame 0
        last_aligned_decoded = 0
        last_aligned_emitted = 0
        decoder_state = None  # Force fresh decode from start
        print(f"[DEBUG STREAM] debug_include_warmup=True: outputting ALL frames from 0, ignoring cached decoder_state", file=sys.stderr)
    else:
        last_aligned_decoded = max(0, (start_step + 1) - max_delay)  # First frame to decode in streaming
        last_aligned_emitted = last_aligned_decoded + 1  # Skip transition frame, output from frame with more user influence

    # DEBUG: Log decoder_state info at start
    print(f"[DEBUG STREAM] decoder_state provided: {decoder_state is not None}", file=sys.stderr)
    if decoder_state is not None:
        print(f"[DEBUG STREAM] decoder_state.kv_cache: {type(decoder_state.kv_cache).__name__ if decoder_state.kv_cache else None}", file=sys.stderr)
        if decoder_state.kv_cache is not None and hasattr(decoder_state.kv_cache, 'get_seq_length'):
            print(f"[DEBUG STREAM] decoder_state.kv_cache.get_seq_length()={decoder_state.kv_cache.get_seq_length()}", file=sys.stderr)
        print(f"[DEBUG STREAM] decoder_state.padding_cache: {decoder_state.padding_cache is not None}", file=sys.stderr)

    # Calculate approximate frame rate for timing
    frame_rate = getattr(runtime, 'frame_rate', 75.0)
    ms_per_frame = 1000.0 / frame_rate

    # Timing accumulators for profiling
    timing_stats = {
        'transformer_step': 0.0,
        'text_sampling': 0.0,
        'machine_process': 0.0,
        'audio_sampling': 0.0,
        'depformer_stages': 0.0,
        'audio_decode': 0.0,
        'cuda_graph_capture': 0.0,
        'step_count': 0,
    }
    first_step_time = None
    cuda_graph_captured = False
    first_chunk_emitted = False
    first_chunk_timing = {}  # Detailed timing for first chunk

    print(f"[DEBUG STREAM] audio_buf.shape={generation.audio_buf.shape}, total_raw_frames={generation.audio_buf.shape[-1]}", file=sys.stderr)
    print(f"[DEBUG STREAM] start_step={start_step}, max_delay={max_delay}", file=sys.stderr)
    print(f"[DEBUG STREAM] last_aligned_decoded={last_aligned_decoded} (decoder state position)", file=sys.stderr)
    print(f"[DEBUG STREAM] last_aligned_emitted={last_aligned_emitted} (emit from frame {last_aligned_emitted + 1})", file=sys.stderr)
    print(f"[TIMING] Starting generation loop: max_context={max_context}, chunk_size={chunk_size}, max_delay={max_delay}, start_step={start_step}", file=sys.stderr)

    # Reset chunk diagnostics and WebM streamer for this generation session
    reset_chunk_diagnostics()
    reset_webm_streamer()

    # Check for debug mode via environment variable
    import os
    debug_dir = os.environ.get("DIA_DEBUG_AUDIO_DIR")
    if debug_dir:
        enable_debug_chunk_save(debug_dir)

    yield StatusEvent(message="Starting generation", progress=0.0)

    try:
        with torch.inference_mode():
            for offset in range(max_context):
                step_start = time.time()

                if first_step_time is None:
                    first_step_time = step_start
                    print(f"[TIMING] First step started at {(step_start - generation_start_time)*1000:.0f}ms", file=sys.stderr)

                # Log first 5 steps for debugging first chunk latency
                log_this_step = offset < 5

                if use_torch_compile:
                    torch.compiler.cudagraph_mark_step_begin()

                t = start_step + offset

                if eos_cutoff is not None and t >= eos_cutoff:
                    print(f"[TIMING] Breaking: eos_cutoff reached at step {t} (eos_cutoff={eos_cutoff})", file=sys.stderr)
                    break
                if t + 1 >= audio_buf.shape[-1]:
                    print(f"[TIMING] Breaking: audio_buf limit reached at step {t} (buf_size={audio_buf.shape[-1]})", file=sys.stderr)
                    break

                generation.reset_dep_cache()
                positions.fill_(t)
                _fill_audio_channels(step_tokens, audio_buf, delay_tensor, t, token_ids.audio_bos)

                if branches > 1:
                    step_tokens[1:, 0, 0] = token_ids.zero
                    step_tokens[1:, 1, 0] = token_ids.pad

                # Transformer step - use CUDA graph if enabled
                t0 = time.time()
                is_graph_capture = (use_graph and transformer_capture is None)

                if transformer_needs_compiling or not use_graph:
                    if transformer_needs_compiling:
                        transformer_step = torch.compile(
                            runtime.transformer_step,
                            dynamic=True,
                            mode="max-autotune-no-cudagraphs",
                        )
                        transformer_needs_compiling = False
                    hidden_t = _execute_transformer_step(
                        step_tokens,
                        positions_view,
                        generation,
                        transformer_step,
                        buffers,
                    )
                else:
                    transformer_capture, dep_captures = _execute_transformer_graph(
                        runtime=runtime,
                        step_tokens=step_tokens,
                        positions_view=positions_view,
                        branches=branches,
                        generation=generation,
                        transformer_step=transformer_step,
                        buffers=buffers,
                        transformer_capture=transformer_capture,
                        dep_captures=dep_captures,
                    )
                    hidden_t = transformer_capture[1]

                # EXPLICIT SYNC: Wait for transformer to complete before timing text sampling
                # This ensures accurate timing attribution (otherwise .item() waits for transformer)
                if is_cuda:
                    torch.cuda.synchronize()

                t1 = time.time()
                transformer_elapsed = t1 - t0
                if is_graph_capture:
                    timing_stats['cuda_graph_capture'] += transformer_elapsed
                    print(f"[TIMING] CUDA graph capture took {transformer_elapsed*1000:.0f}ms", file=sys.stderr)
                    cuda_graph_captured = True
                else:
                    timing_stats['transformer_step'] += transformer_elapsed
                    if log_this_step:
                        print(f"[TIMING] Step {offset}: transformer={transformer_elapsed*1000:.1f}ms", file=sys.stderr)

                # Text token sampling (now accurately timed - transformer already synced)
                t0 = time.time()
                guided_text = apply_classifier_guidance(buffers.text, cfg_active, config.cfg_scale, config.cfg_filter_k)
                if guided_text.shape[0] > 1:
                    guided_text = guided_text[:1]

                text_token = sample_token_fn(
                    guided_text,
                    temp=config.text.temperature,
                    top_k=config.text.top_k,
                ).item()
                t1 = time.time()
                timing_stats['text_sampling'] += (t1 - t0)

                t0 = time.time()
                main_token, aux_token, _ = runtime.machine.process(t, state, text_token)
                second_token = aux_token if aux_token != -1 else token_ids.pad
                t1 = time.time()
                timing_stats['machine_process'] += (t1 - t0)

                if first_word_frame is None and main_token == token_ids.new_word:
                    first_word_frame = t - config.initial_padding
                    print(f"[TIMING] First word detected at step {t}, first_word_frame={first_word_frame}", file=sys.stderr)

                step_tokens[:, 0, 0] = main_token
                step_tokens[:, 1, 0] = second_token

                # Audio token sampling (CB0)
                t0 = time.time()
                guided_cb0 = apply_classifier_guidance(buffers.cb0, cfg_active, config.cfg_scale, config.cfg_filter_k)
                if guided_cb0.shape[0] > 1:
                    guided_cb0 = guided_cb0[:1]
                masked_cb0 = mask_audio_logits(guided_cb0, token_ids.audio_pad, token_ids.audio_bos)
                codebook_token = sample_audio_logits_fn(masked_cb0, config.audio.temperature, config.audio.top_k)
                audio_buf[:, 0, t + 1] = codebook_token

                # EXPLICIT SYNC: Ensure CB0 sampling is complete before timing
                if is_cuda:
                    torch.cuda.synchronize()

                t1 = time.time()
                timing_stats['audio_sampling'] += (t1 - t0)

                prev_audio = codebook_token.expand(branches)
                main_tokens.fill_(main_token)
                aux_tokens.fill_(second_token)

                # Depformer stages - use CUDA graph if enabled
                t0 = time.time()
                for stage in range(runtime.model.depformer.num_depth):
                    if use_graph and dep_captures is not None:
                        if depformer_needs_compiling[stage]:
                            runtime.model.depformer._forward_stage = torch.compile(
                                runtime.model.depformer._forward_stage,
                                dynamic=True,
                                mode="max-autotune-no-cudagraphs",
                            )
                            depformer_needs_compiling[stage] = False
                            _execute_depformer_stage(
                                stage_index=stage,
                                prev_audio=prev_audio,
                                hidden_t=hidden_t,
                                generation=generation,
                                depformer_step=depformer_step,
                                main_tokens=main_tokens,
                                second_tokens=aux_tokens,
                                buffers=buffers,
                            )
                        else:
                            dep_captures[stage] = _execute_depformer_graph(
                                stage=stage,
                                prev_audio=prev_audio,
                                hidden_t=hidden_t,
                                generation=generation,
                                depformer_step=depformer_step,
                                main_tokens=main_tokens,
                                aux_tokens=aux_tokens,
                                buffers=buffers,
                                capture=dep_captures[stage],
                            )
                    else:
                        if depformer_needs_compiling[stage]:
                            runtime.model.depformer._forward_stage = torch.compile(
                                runtime.model.depformer._forward_stage,
                                dynamic=True,
                                mode="max-autotune-no-cudagraphs",
                            )
                            depformer_needs_compiling[stage] = False

                        _execute_depformer_stage(
                            stage_index=stage,
                            prev_audio=prev_audio,
                            hidden_t=hidden_t,
                            generation=generation,
                            depformer_step=depformer_step,
                            main_tokens=main_tokens,
                            second_tokens=aux_tokens,
                            buffers=buffers,
                        )

                    dep_logits = apply_classifier_guidance(
                        buffers.dep[stage], cfg_active, config.cfg_scale, config.cfg_filter_k
                    )
                    if dep_logits.shape[0] > 1:
                        dep_logits = dep_logits[:1]

                    stage_token = sample_audio_logits_fn(
                        dep_logits,
                        config.audio.temperature,
                        config.audio.top_k,
                    )
                    audio_buf[:, stage + 1, t + 1] = stage_token
                    prev_audio = stage_token.expand(branches)

                # EXPLICIT SYNC: Wait for depformer to complete for accurate timing
                if is_cuda:
                    torch.cuda.synchronize()

                t1 = time.time()
                depformer_elapsed = t1 - t0
                timing_stats['depformer_stages'] += depformer_elapsed

                if log_this_step:
                    step_total = (t1 - step_start) * 1000
                    print(f"[TIMING] Step {offset}: depformer={depformer_elapsed*1000:.1f}ms, step_total={step_total:.1f}ms @ {(t1 - generation_start_time)*1000:.0f}ms", file=sys.stderr)

                last_step = t
                timing_stats['step_count'] += 1

                if eos_cutoff is None and state.end_step is not None:
                    eos_cutoff = state.end_step + flush_tail
                    print(f"[TIMING] EOS detected at step {t}, eos_cutoff set to {eos_cutoff}", file=sys.stderr)

                # Skip initial frames before first_word_frame (warmup/padding frames)
                # This matches the crop behavior in non-streaming generate()
                if not first_word_frame_applied and first_word_frame is not None:
                    skip_to = max(first_word_frame, 0)
                    if skip_to > last_aligned_emitted:
                        print(f"[TIMING] Skipping initial frames: last_aligned_emitted {last_aligned_emitted} -> {skip_to}", file=sys.stderr)
                        last_aligned_emitted = skip_to
                    first_word_frame_applied = True

                # Check if we have enough NEW aligned frames for a chunk
                # We check against last_aligned_emitted (output position), not last_aligned_decoded
                current_aligned_len = max(0, (t + 1) - max_delay)
                new_frames_to_output = current_aligned_len - last_aligned_emitted
                should_emit_chunk = new_frames_to_output >= chunk_size

                # Also emit on EOS if we have enough frames
                at_end = (eos_cutoff is not None and t + 1 >= eos_cutoff) or (t + 2 >= audio_buf.shape[-1])
                if at_end and new_frames_to_output >= min_chunk:
                    should_emit_chunk = True

                if should_emit_chunk:
                    # ASYNC DECODE PIPELINE:
                    # 1. First, yield any pending chunk from PREVIOUS async decode
                    # 2. Decode from last_aligned_decoded (decoder position) to current
                    # 3. Slice waveform to only output from last_aligned_emitted (skip transition frames)
                    # 4. Store waveform for next iteration

                    if log_this_step:
                        print(f"[TIMING] Step {offset}: should_emit_chunk=True, new_output={new_frames_to_output}, pending={pending_decode is not None} @ {(time.time() - generation_start_time)*1000:.0f}ms", file=sys.stderr)

                    # Step 1: Yield pending chunk from previous async decode
                    if pending_decode is not None:
                        prev_waveform, prev_idx, prev_ts = pending_decode
                        try:
                            if decode_stream is not None:
                                decode_stream.synchronize()

                            t0 = time.time()
                            if prev_waveform.numel() > 0:
                                audio_bytes = _encode_webm_chunk(prev_waveform, runtime.mimi.sample_rate)
                                chunk_duration_ms = (prev_waveform.numel() / runtime.mimi.sample_rate) * 1000

                                if not first_chunk_emitted:
                                    print(f"[TIMING] First audio chunk at {(time.time() - generation_start_time)*1000:.0f}ms (chunk_idx={prev_idx}, bytes={len(audio_bytes)})", file=sys.stderr)
                                    first_chunk_emitted = True

                                yield AudioChunkEvent(
                                    audio_data=audio_bytes,
                                    chunk_index=prev_idx,
                                    timestamp_ms=prev_ts,
                                    duration_ms=chunk_duration_ms,
                                )
                                total_audio_ms += chunk_duration_ms

                            t1 = time.time()
                            timing_stats['audio_decode'] += (t1 - t0)
                        except Exception as e:
                            if logger:
                                logger.event(f"Chunk encode error: {e}")
                            yield ErrorEvent(error=f"Chunk encode error: {e}")
                            return
                        pending_decode = None

                    # Step 2: Prepare new chunk for async decode
                    # We decode from last_aligned_decoded (where decoder left off) to current
                    # But only OUTPUT audio starting from last_aligned_emitted (skip transition frames)
                    current_frame = t + 1

                    full_tokens = audio_buf[0, :, :current_frame + 1]
                    aligned_full = undelay_frames(
                        full_tokens,
                        runtime.audio_delays,
                        token_ids.audio_pad
                    ).unsqueeze(0)

                    current_aligned_end = aligned_full.shape[-1]

                    # Only proceed if we have new frames to decode
                    if current_aligned_end > last_aligned_decoded:
                        # Decode all frames from decoder position to current (includes transition frames)
                        frames_to_decode = aligned_full[:, :, last_aligned_decoded:current_aligned_end]
                        num_frames_decoded = frames_to_decode.shape[-1]

                        # Calculate how many of these frames should be output vs discarded
                        # Transition frames are between last_aligned_decoded and last_aligned_emitted
                        # Cap at num_frames_decoded to avoid skipping more than we have
                        frames_to_skip = max(0, min(last_aligned_emitted - last_aligned_decoded, num_frames_decoded))
                        frames_to_output = num_frames_decoded - frames_to_skip

                        # Debug: log decode range for first few chunks
                        if chunk_index < 5:
                            print(f"[DEBUG DECODE] chunk {chunk_index}: t={t}, current_frame={current_frame}", file=sys.stderr)
                            print(f"[DEBUG DECODE] chunk {chunk_index}: aligned_full.shape={aligned_full.shape}, current_aligned_end={current_aligned_end}", file=sys.stderr)
                            print(f"[DEBUG DECODE] chunk {chunk_index}: decoding aligned[{last_aligned_decoded}:{current_aligned_end}] = {num_frames_decoded} frames", file=sys.stderr)
                            print(f"[DEBUG DECODE] chunk {chunk_index}: frames_to_skip={frames_to_skip} (transition), frames_to_output={frames_to_output}", file=sys.stderr)
                            print(f"[DEBUG DECODE] chunk {chunk_index}: decoder_state.kv_cache={type(decoder_state.kv_cache).__name__ if decoder_state and decoder_state.kv_cache else None}", file=sys.stderr)

                        if frames_to_decode.shape[-1] > 0:
                            try:
                                t0 = time.time()
                                is_first_decode = (chunk_index == 0)

                                # Decode the full range (including transition frames)
                                if decode_stream is not None:
                                    with torch.cuda.stream(decode_stream):
                                        full_waveform, decoder_state = decode_audio_streaming(
                                            runtime, frames_to_decode, decoder_state
                                        )
                                else:
                                    full_waveform, decoder_state = decode_audio_streaming(
                                        runtime, frames_to_decode, decoder_state
                                    )

                                # Slice the waveform to skip transition frames
                                # Each frame = samples_per_frame samples
                                samples_per_frame = runtime.mimi.samples_per_frame
                                samples_to_skip = frames_to_skip * samples_per_frame
                                output_waveform = full_waveform[samples_to_skip:]

                                # Prime the encoder with transition audio to prevent click at start
                                if samples_to_skip > 0 and chunk_index == 0:
                                    prime_samples = min(2000, samples_to_skip)
                                    prime_start = samples_to_skip - prime_samples
                                    prime_audio = full_waveform[prime_start:samples_to_skip]
                                    streamer = get_webm_streamer(runtime.mimi.sample_rate)
                                    streamer.prime_encoder(prime_audio.detach().cpu().numpy())

                                # Debug: log waveform sizes
                                if chunk_index < 5:
                                    full_samples = full_waveform.numel()
                                    output_samples = output_waveform.numel()
                                    full_ms = (full_samples / runtime.mimi.sample_rate) * 1000
                                    output_ms = (output_samples / runtime.mimi.sample_rate) * 1000
                                    print(f"[DEBUG DECODE] chunk {chunk_index}: full waveform = {full_samples} samples = {full_ms:.1f}ms", file=sys.stderr)
                                    print(f"[DEBUG DECODE] chunk {chunk_index}: skipping {samples_to_skip} samples, output = {output_samples} samples = {output_ms:.1f}ms", file=sys.stderr)

                                # Update decoder position
                                last_aligned_decoded = current_aligned_end

                                # Only emit if we have output audio
                                if output_waveform.numel() > 0:
                                    if decode_stream is not None:
                                        pending_decode = (output_waveform, chunk_index, total_audio_ms)
                                        chunk_index += 1
                                    else:
                                        audio_bytes = _encode_webm_chunk(output_waveform, runtime.mimi.sample_rate)
                                        chunk_duration_ms = (output_waveform.numel() / runtime.mimi.sample_rate) * 1000

                                        if not first_chunk_emitted:
                                            print(f"[TIMING] First audio chunk at {(time.time() - generation_start_time)*1000:.0f}ms (chunk_idx={chunk_index}, bytes={len(audio_bytes)})", file=sys.stderr)
                                            first_chunk_emitted = True

                                        yield AudioChunkEvent(
                                            audio_data=audio_bytes,
                                            chunk_index=chunk_index,
                                            timestamp_ms=total_audio_ms,
                                            duration_ms=chunk_duration_ms,
                                        )
                                        total_audio_ms += chunk_duration_ms
                                        chunk_index += 1

                                    # Update output position
                                    last_aligned_emitted = current_aligned_end

                                t1 = time.time()
                                if is_first_decode:
                                    print(f"[TIMING] First decode took {(t1-t0)*1000:.1f}ms (async={decode_stream is not None}) @ {(t1 - generation_start_time)*1000:.0f}ms", file=sys.stderr)

                            except Exception as e:
                                if logger:
                                    logger.event(f"Chunk decode error: {e}")
                                yield ErrorEvent(error=f"Chunk decode error: {e}")
                                return

                # Emit status periodically
                if (offset + 1) % (streaming_config.emit_status_every * chunk_size) == 0:
                    progress = min(1.0, (offset + 1) / max_context)
                    yield StatusEvent(
                        message=f"Generating audio ({offset + 1}/{max_context} steps)",
                        progress=progress
                    )

        # First, yield any pending async decode from the loop
        if pending_decode is not None:
            prev_waveform, prev_idx, prev_ts = pending_decode
            try:
                if decode_stream is not None:
                    decode_stream.synchronize()

                if prev_waveform.numel() > 0:
                    audio_bytes = _encode_webm_chunk(prev_waveform, runtime.mimi.sample_rate)
                    chunk_duration_ms = (prev_waveform.numel() / runtime.mimi.sample_rate) * 1000

                    yield AudioChunkEvent(
                        audio_data=audio_bytes,
                        chunk_index=prev_idx,
                        timestamp_ms=prev_ts,
                        duration_ms=chunk_duration_ms,
                    )
                    total_audio_ms += chunk_duration_ms

            except Exception as e:
                if logger:
                    logger.event(f"Pending chunk encode error: {e}")
                yield ErrorEvent(error=f"Pending chunk encode error: {e}")
                return
            pending_decode = None

        # Handle any remaining frames at end of generation
        # Use same two-variable logic: decode from last_aligned_decoded, output from last_aligned_emitted
        print(f"[DEBUG END] last_step={last_step}, last_aligned_decoded={last_aligned_decoded}, last_aligned_emitted={last_aligned_emitted}", file=sys.stderr)
        if last_step >= 0:
            remaining_end = min(last_step + 2, audio_buf.shape[-1])
            print(f"[DEBUG END] remaining_end={remaining_end}", file=sys.stderr)

            if remaining_end > 0:
                full_tokens = audio_buf[0, :, :remaining_end]
                aligned_full = undelay_frames(
                    full_tokens,
                    runtime.audio_delays,
                    token_ids.audio_pad
                ).unsqueeze(0)

                current_aligned_end = aligned_full.shape[-1]
                print(f"[DEBUG END] current_aligned_end={current_aligned_end}, full_tokens.shape={full_tokens.shape}", file=sys.stderr)

                # Decode any frames we haven't decoded yet
                if current_aligned_end > last_aligned_decoded:
                    frames_to_decode = aligned_full[:, :, last_aligned_decoded:current_aligned_end]
                    num_frames_decoded = frames_to_decode.shape[-1]

                    # Calculate frames to skip (transition frames)
                    # Cap at num_frames_decoded to avoid skipping more than we have
                    frames_to_skip = max(0, min(last_aligned_emitted - last_aligned_decoded, num_frames_decoded))
                    frames_to_output = num_frames_decoded - frames_to_skip
                    print(f"[DEBUG END] num_frames_decoded={num_frames_decoded}, frames_to_skip={frames_to_skip}, frames_to_output={frames_to_output}", file=sys.stderr)

                    if frames_to_decode.shape[-1] > 0:
                        try:
                            full_waveform, decoder_state = decode_audio_streaming(
                                runtime, frames_to_decode, decoder_state
                            )

                            # Slice to skip transition frames
                            samples_per_frame = runtime.mimi.samples_per_frame
                            samples_to_skip = frames_to_skip * samples_per_frame
                            output_waveform = full_waveform[samples_to_skip:]
                            print(f"[DEBUG END] full_waveform.numel()={full_waveform.numel()}, samples_to_skip={samples_to_skip}, output_waveform.numel()={output_waveform.numel()}", file=sys.stderr)

                            if output_waveform.numel() > 0:
                                audio_bytes = _encode_webm_chunk(output_waveform, runtime.mimi.sample_rate)
                                chunk_duration_ms = (output_waveform.numel() / runtime.mimi.sample_rate) * 1000
                                print(f"[DEBUG END] Emitting final chunk: {chunk_duration_ms:.1f}ms", file=sys.stderr)

                                yield AudioChunkEvent(
                                    audio_data=audio_bytes,
                                    chunk_index=chunk_index,
                                    timestamp_ms=total_audio_ms,
                                    duration_ms=chunk_duration_ms,
                                )
                                total_audio_ms += chunk_duration_ms
                                chunk_index += 1
                            else:
                                print(f"[DEBUG END] No output audio (all transition frames)", file=sys.stderr)

                        except Exception as e:
                            if logger:
                                logger.event(f"Final chunk decode error: {e}")
                            yield ErrorEvent(error=f"Final chunk decode error: {e}")
                            return
                else:
                    print(f"[DEBUG END] No new frames to decode (current_aligned_end={current_aligned_end} <= last_aligned_decoded={last_aligned_decoded})", file=sys.stderr)

        # Save concatenated debug audio if enabled
        save_debug_concatenated()

        # Finalize the WebM stream to flush any remaining audio in the Opus encoder
        streamer = get_webm_streamer(runtime.mimi.sample_rate)
        final_bytes = streamer.finalize()
        if final_bytes and len(final_bytes) > 0:
            print(f"[DEBUG END] Finalized WebM stream, got {len(final_bytes)} bytes", file=sys.stderr)
            yield AudioChunkEvent(
                audio_data=final_bytes,
                chunk_index=chunk_index,
                timestamp_ms=total_audio_ms,
                duration_ms=0,  # Duration unknown for finalization data
            )
            chunk_index += 1

        # Emit completion event
        yield CompleteEvent(
            total_chunks=chunk_index,
            total_duration_ms=total_audio_ms,
        )

        # Print timing summary (with explicit CUDA syncs for accurate attribution)
        total_elapsed = time.time() - generation_start_time
        step_count = timing_stats['step_count']
        print(f"\n[TIMING SUMMARY - ACCURATE] Total: {total_elapsed*1000:.0f}ms, Steps: {step_count}, Audio: {total_audio_ms:.0f}ms", file=sys.stderr)
        print(f"  CUDA graph capture:  {timing_stats['cuda_graph_capture']*1000:.0f}ms (one-time)", file=sys.stderr)
        print(f"  Transformer+sync:    {timing_stats['transformer_step']*1000:.0f}ms ({timing_stats['transformer_step']/max(step_count,1)*1000:.1f}ms/step)", file=sys.stderr)
        print(f"  Text sampling+sync:  {timing_stats['text_sampling']*1000:.0f}ms ({timing_stats['text_sampling']/max(step_count,1)*1000:.1f}ms/step)", file=sys.stderr)
        print(f"  Machine process:     {timing_stats['machine_process']*1000:.0f}ms ({timing_stats['machine_process']/max(step_count,1)*1000:.1f}ms/step)", file=sys.stderr)
        print(f"  Audio CB0+sync:      {timing_stats['audio_sampling']*1000:.0f}ms ({timing_stats['audio_sampling']/max(step_count,1)*1000:.1f}ms/step)", file=sys.stderr)
        print(f"  Depformer+sync:      {timing_stats['depformer_stages']*1000:.0f}ms ({timing_stats['depformer_stages']/max(step_count,1)*1000:.1f}ms/step)", file=sys.stderr)
        print(f"  Audio decode:        {timing_stats['audio_decode']*1000:.0f}ms", file=sys.stderr)
        accounted = sum(timing_stats.values()) - timing_stats['step_count']
        print(f"  Other/overhead:      {(total_elapsed - accounted)*1000:.0f}ms", file=sys.stderr)

        if logger:
            elapsed = time.time() - generation_start_time
            logger.event(f"Streaming generation finished in {elapsed:.2f}s, {chunk_index} chunks, {total_audio_ms:.0f}ms audio")

    except Exception as e:
        yield ErrorEvent(error=str(e))
        raise

def warmup_with_prefix(
    runtime: RuntimeContext,
    plan: PrefixPlan,
    state: State,
    generation: GenerationState,
) -> int:
    step_tokens = generation.step_tokens
    model_state = generation.decode
    branches = step_tokens.shape[0]
    device = runtime.device
    tokens = plan.aligned_tokens.to(device)
    new_word_steps = set(plan.new_word_steps)
    positions = torch.empty(1, 1, dtype=torch.long, device=device)

    with torch.inference_mode():
        for t in range(plan.aligned_frames):
            positions.fill_(t)
            channels = tokens.shape[0]
            for cb in range(channels):
                delay = runtime.audio_delays[cb] if cb < len(runtime.audio_delays) else 0
                idx = t - delay
                value = tokens[cb, idx] if idx >= 0 else runtime.constants.audio_bos
                step_tokens[:, 2 + cb, 0] = value
            hidden, text_logits, cb0_logits, present = runtime.model.transformer.forward_step(
                step_tokens,
                positions.expand(branches, -1),
                model_state.transformer,
            )
            model_state.transformer = present

            forced = runtime.constants.new_word if t in new_word_steps else runtime.constants.pad
            main_token, aux_token, _ = runtime.machine.process(t, state, forced, is_forced=True)
            second_token = runtime.constants.pad if aux_token == -1 else aux_token
            step_tokens[0, 0, 0] = main_token
            step_tokens[0, 1, 0] = second_token
            if branches > 1:
                step_tokens[1:, 0, 0] = runtime.constants.zero
                step_tokens[1:, 1, 0] = runtime.constants.pad

    return max(plan.aligned_frames - 1, 0)
__all__ = [
    "build_initial_state",
    "run_generation_loop",
    "run_streaming_generation_loop",
    "run_seed_warmup",
    "decode_audio",
    "warmup_with_prefix",
    "GenerationState",
    "reset_chunk_diagnostics",
    "enable_debug_chunk_save",
    "disable_debug_chunk_save",
    "save_debug_concatenated",
    "WebMOpusStreamer",
    "get_webm_streamer",
    "reset_webm_streamer",
]
