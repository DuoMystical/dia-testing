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
    lookahead_tokens: torch.Tensor = None,
):
    """Decode audio tokens to waveform with optional lookahead for seamless streaming.

    Uses the MimiCodec's overlap decode to ensure smooth audio transitions
    between chunks without boundary artifacts.

    Args:
        runtime: Runtime context containing the mimi codec
        tokens: Audio tokens to decode (batch, codebooks, frames)
        decoder_state: Previous StreamingDecoderState, or None for first chunk
        lookahead_tokens: Optional lookahead tokens for forward context.
                         If provided, these are appended to tokens but only
                         tokens' audio is output (lookahead provides forward context).

    Returns:
        Tuple of (waveform, new_decoder_state)
        - waveform: 1D tensor of audio samples (for tokens only, not lookahead)
        - new_decoder_state: State to pass to next call
    """
    if tokens.shape[-1] == 0:
        return torch.zeros(0, device=runtime.device), decoder_state

    with torch.inference_mode():
        if lookahead_tokens is not None and lookahead_tokens.shape[-1] > 0:
            # Overlap decode: combine tokens with lookahead for forward context
            combined = torch.cat([tokens, lookahead_tokens], dim=-1)
            output_frames = tokens.shape[-1]
            pcm, new_state = runtime.mimi.decode_with_lookahead(
                combined.to(runtime.device),
                output_frames=output_frames,
                decoder_state=decoder_state,
            )
        else:
            # Regular decode without lookahead
            pcm, new_state = runtime.mimi.decode_with_state(
                tokens.to(runtime.device),
                decoder_state=decoder_state,
            )
        return pcm[0, 0], new_state


# Module-level tracking for cross-chunk boundary analysis
_last_chunk_end_samples = None
_chunk_diagnostics_enabled = True
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


def _encode_opus_chunk(
    waveform: torch.Tensor,
    sample_rate: int,
) -> bytes:
    """Encode a waveform tensor to Opus/OGG bytes.

    Opus is designed for streaming and handles packet boundaries gracefully,
    eliminating the clicks/pops that occur with WAV when samples don't end at zero.
    """
    import io
    import numpy as np
    import soundfile as sf
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

    # Encode to Opus in OGG container
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='OGG', subtype='OPUS')
    opus_bytes = buffer.getvalue()

    if _chunk_diagnostics_enabled:
        print(f"  [OPUS] samples={len(audio_np)}, sample_rate={sample_rate}Hz, size={len(opus_bytes)} bytes", file=sys.stderr)
        print(f"  [OPUS] first 5 samples: {audio_np[:5].tolist()}", file=sys.stderr)
        print(f"  [OPUS] last 5 samples: {audio_np[-5:].tolist()}", file=sys.stderr)

    return opus_bytes


def run_seed_warmup(
    runtime: RuntimeContext,
    generation: GenerationState,
    config: GenerationConfig,
    num_steps: int,
    warmup_state: Optional[State] = None,
) -> GenerationState:
    """
    Run warmup steps to establish seed-dependent state.

    This runs num_steps of generation with a fixed warmup phrase, building up
    the KV cache and audio buffer. The warmup phrase ensures consistent behavior
    across different user texts when using the same seed.

    Args:
        runtime: Runtime context
        generation: Initial generation state
        config: Generation configuration
        num_steps: Number of warmup steps (should be >= max_delay)
        warmup_state: State with warmup phrase entries (created by caller)

    Returns:
        The generation state after warmup, ready for caching
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

    print(f"[WARMUP] Running {num_steps} warmup steps for seed caching", file=sys.stderr)

    with torch.inference_mode():
        for t in range(num_steps):
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

    print(f"[WARMUP] Completed {num_steps} warmup steps", file=sys.stderr)
    return generation


def run_streaming_generation_loop(
    runtime: RuntimeContext,
    *,
    state: State,
    generation: GenerationState,
    config: GenerationConfig,
    streaming_config: StreamingConfig,
    start_step: int = 0,
    logger: RuntimeLogger | None = None,
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

    # Streaming state
    chunk_size = streaming_config.chunk_size_frames
    min_chunk = streaming_config.min_chunk_frames
    chunk_index = 0
    total_audio_ms = 0.0
    generation_start_time = time.time()

    # Decoder state for maintaining continuity between audio chunks
    # This preserves the Mimi decoder's internal state (key-value cache) across chunks,
    # eliminating boundary artifacts that occur when decoding chunks independently
    decoder_state = None

    # Track aligned frames we've already emitted (after undelay)
    # undelay output length = input_length - max_delay
    # Skip warmup frames - they were used to prime the pipeline and should be clipped
    # warmup produces (start_step - max_delay) aligned frames that we don't emit
    last_aligned_emitted = max(0, start_step - max_delay)

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

    print(f"[TIMING] Starting generation loop: max_context={max_context}, chunk_size={chunk_size}, max_delay={max_delay}, start_step={start_step}, last_aligned_emitted={last_aligned_emitted}", file=sys.stderr)

    # Reset chunk diagnostics for this generation session
    reset_chunk_diagnostics()

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
                if is_graph_capture:
                    timing_stats['cuda_graph_capture'] += (t1 - t0)
                    print(f"[TIMING] CUDA graph capture took {(t1-t0)*1000:.0f}ms", file=sys.stderr)
                    cuda_graph_captured = True
                else:
                    timing_stats['transformer_step'] += (t1 - t0)

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
                timing_stats['depformer_stages'] += (t1 - t0)

                last_step = t
                timing_stats['step_count'] += 1

                if eos_cutoff is None and state.end_step is not None:
                    eos_cutoff = state.end_step + flush_tail
                    print(f"[TIMING] EOS detected at step {t}, eos_cutoff set to {eos_cutoff}", file=sys.stderr)

                # NOTE: We no longer skip based on first_word_frame for streaming.
                # With seed caching, the warmup frames are already accounted for in last_aligned_emitted
                # (initialized to start_step - max_delay). Skipping based on first_word_frame caused
                # the 450ms delay bug because it would wait for the first new_word token.
                first_word_frame_applied = True

                # Check if we have enough NEW aligned frames for a chunk
                # aligned length = (t + 1) - max_delay, we've emitted last_aligned_emitted
                current_aligned_len = max(0, (t + 1) - max_delay)
                new_aligned_frames = current_aligned_len - last_aligned_emitted
                should_emit_chunk = new_aligned_frames >= chunk_size

                # Also emit on EOS if we have enough frames
                at_end = (eos_cutoff is not None and t + 1 >= eos_cutoff) or (t + 2 >= audio_buf.shape[-1])
                if at_end and new_aligned_frames >= min_chunk:
                    should_emit_chunk = True

                if should_emit_chunk:
                    # IMMEDIATE DECODE: No lookahead, no pending - decode and emit right away
                    # This minimizes latency to first audio chunk
                    current_frame = t + 1
                    is_first_chunk = not first_chunk_emitted

                    # Undelay the entire buffer up to current position
                    t_undelay_start = time.time()
                    full_tokens = audio_buf[0, :, :current_frame + 1]
                    aligned_full = undelay_frames(
                        full_tokens,
                        runtime.audio_delays,
                        token_ids.audio_pad
                    ).unsqueeze(0)
                    t_undelay_end = time.time()

                    current_aligned_end = aligned_full.shape[-1]

                    if current_aligned_end > last_aligned_emitted:
                        aligned_chunk = aligned_full[:, :, last_aligned_emitted:current_aligned_end]

                        if aligned_chunk.shape[-1] > 0:
                            try:
                                t_decode_start = time.time()
                                # Decode immediately without lookahead
                                chunk_waveform, decoder_state = decode_audio_streaming(
                                    runtime, aligned_chunk, decoder_state
                                )
                                t_decode_end = time.time()

                                if chunk_waveform.numel() > 0:
                                    t_encode_start = time.time()
                                    audio_bytes = _encode_opus_chunk(chunk_waveform, runtime.mimi.sample_rate)
                                    t_encode_end = time.time()
                                    chunk_duration_ms = (chunk_waveform.numel() / runtime.mimi.sample_rate) * 1000

                                    yield AudioChunkEvent(
                                        audio_data=audio_bytes,
                                        chunk_index=chunk_index,
                                        timestamp_ms=total_audio_ms,
                                    )

                                    # Print detailed timing for first chunk
                                    if is_first_chunk:
                                        first_chunk_emitted = True
                                        total_time = time.time() - generation_start_time
                                        steps_run = offset + 1
                                        print(f"[TIMING] === FIRST CHUNK BREAKDOWN ===", file=sys.stderr)
                                        print(f"[TIMING]   Step: {t} (offset={offset}, steps_run={steps_run})", file=sys.stderr)
                                        print(f"[TIMING]   Aligned frames emitted: {last_aligned_emitted} -> {current_aligned_end}", file=sys.stderr)
                                        print(f"[TIMING]   --- Generation steps ({steps_run} steps) ---", file=sys.stderr)
                                        print(f"[TIMING]   Transformer: {timing_stats['transformer_step']*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   Text sampling: {timing_stats['text_sampling']*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   State machine: {timing_stats['machine_process']*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   Audio sampling: {timing_stats['audio_sampling']*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   Depformer: {timing_stats['depformer_stages']*1000:.1f}ms", file=sys.stderr)
                                        if timing_stats['cuda_graph_capture'] > 0:
                                            print(f"[TIMING]   CUDA graph capture: {timing_stats['cuda_graph_capture']*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   --- Audio encoding ---", file=sys.stderr)
                                        print(f"[TIMING]   Undelay: {(t_undelay_end - t_undelay_start)*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   Mimi decode: {(t_decode_end - t_decode_start)*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   Opus encode: {(t_encode_end - t_encode_start)*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   --- Summary ---", file=sys.stderr)
                                        print(f"[TIMING]   Chunk audio duration: {chunk_duration_ms:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING]   TOTAL time to first chunk: {total_time*1000:.1f}ms", file=sys.stderr)
                                        print(f"[TIMING] ==============================", file=sys.stderr)

                                    total_audio_ms += chunk_duration_ms
                                    chunk_index += 1

                                timing_stats['audio_decode'] += (t_decode_end - t_decode_start)

                            except Exception as e:
                                if logger:
                                    logger.event(f"Chunk decode error: {e}")
                                yield ErrorEvent(error=f"Chunk decode error: {e}")
                                return

                        last_aligned_emitted = current_aligned_end

                # Emit status periodically
                if (offset + 1) % (streaming_config.emit_status_every * chunk_size) == 0:
                    progress = min(1.0, (offset + 1) / max_context)
                    yield StatusEvent(
                        message=f"Generating audio ({offset + 1}/{max_context} steps)",
                        progress=progress
                    )

        # Handle any remaining frames at end of generation
        if last_step >= 0:
            remaining_end = min(last_step + 2, audio_buf.shape[-1])

            if remaining_end > 0:
                full_tokens = audio_buf[0, :, :remaining_end]
                aligned_full = undelay_frames(
                    full_tokens,
                    runtime.audio_delays,
                    token_ids.audio_pad
                ).unsqueeze(0)

                current_aligned_end = aligned_full.shape[-1]

                if current_aligned_end > last_aligned_emitted:
                    aligned_remaining = aligned_full[:, :, last_aligned_emitted:current_aligned_end]

                    if aligned_remaining.shape[-1] > 0:
                        try:
                            final_waveform, decoder_state = decode_audio_streaming(
                                runtime, aligned_remaining, decoder_state
                            )

                            if final_waveform.numel() > 0:
                                audio_bytes = _encode_opus_chunk(final_waveform, runtime.mimi.sample_rate)
                                chunk_duration_ms = (final_waveform.numel() / runtime.mimi.sample_rate) * 1000

                                yield AudioChunkEvent(
                                    audio_data=audio_bytes,
                                    chunk_index=chunk_index,
                                    timestamp_ms=total_audio_ms,
                                )
                                total_audio_ms += chunk_duration_ms
                                chunk_index += 1

                        except Exception as e:
                            if logger:
                                logger.event(f"Final chunk decode error: {e}")
                            yield ErrorEvent(error=f"Final chunk decode error: {e}")
                            return

        # Save concatenated debug audio if enabled
        save_debug_concatenated()

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
]
