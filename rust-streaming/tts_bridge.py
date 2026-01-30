#!/usr/bin/env python3
"""
TTS Bridge for Dia2 Streaming Server

This script runs as a persistent process, keeping the model loaded between requests.
It reads JSON requests from stdin (one per line) and outputs JSON events to stdout.

Protocol:
- Each request is a single JSON line on stdin
- Each response event is a single JSON line on stdout
- A "complete" or "error" event signals end of a request
"""

import json
import sys
import base64
import signal
import random
from collections import OrderedDict

import torch

# Global model state
_model = None
_device = None
_model_size = None

# LRU seed cache for fast generation with known seeds
# Maps seed -> GenerationState (after warmup)
# Using OrderedDict for LRU behavior - most recently used at end
_seed_cache = OrderedDict()
_MAX_SEED_CACHE_SIZE = 10  # Limit to avoid OOM


def _cache_get(seed: int):
    """Get from cache with LRU update (move to end if exists)."""
    if seed in _seed_cache:
        _seed_cache.move_to_end(seed)
        return _seed_cache[seed]
    return None


def _cache_put(seed: int, state):
    """Put in cache with LRU eviction."""
    if seed in _seed_cache:
        _seed_cache.move_to_end(seed)
    else:
        if len(_seed_cache) >= _MAX_SEED_CACHE_SIZE:
            # Remove oldest (first) item
            oldest_seed = next(iter(_seed_cache))
            print(f"[CACHE] Evicting seed {oldest_seed} from cache", file=sys.stderr)
            del _seed_cache[oldest_seed]
        _seed_cache[seed] = state
    print(f"[CACHE] Cached seed {seed}, cache size: {len(_seed_cache)}", file=sys.stderr)


def _get_tensor_size_mb(tensor):
    """Get size of a tensor in MB."""
    if tensor is None:
        return 0.0
    return tensor.element_size() * tensor.numel() / (1024 * 1024)


def _log_cache_entry_size(seed, gen_state, state, decoder_state):
    """Log the VRAM size breakdown of a cache entry."""
    sizes = {}

    # GenerationState sizes
    gen_size = 0.0
    if hasattr(gen_state, 'audio_buf'):
        gen_size += _get_tensor_size_mb(gen_state.audio_buf)
        sizes['audio_buf'] = _get_tensor_size_mb(gen_state.audio_buf)
    if hasattr(gen_state, 'step_tokens'):
        gen_size += _get_tensor_size_mb(gen_state.step_tokens)
        sizes['step_tokens'] = _get_tensor_size_mb(gen_state.step_tokens)
    if hasattr(gen_state, 'decode') and hasattr(gen_state.decode, 'kv_cache'):
        kv_size = 0.0
        if gen_state.decode.kv_cache is not None:
            for layer in gen_state.decode.kv_cache:
                if layer is not None:
                    for t in layer:
                        if t is not None:
                            kv_size += _get_tensor_size_mb(t)
        gen_size += kv_size
        sizes['gen_kv_cache'] = kv_size

    # Decoder state sizes
    decoder_size = 0.0
    if decoder_state is not None:
        if decoder_state.kv_cache is not None:
            for layer in decoder_state.kv_cache:
                if layer is not None:
                    for t in layer:
                        if t is not None:
                            decoder_size += _get_tensor_size_mb(t)
            sizes['decoder_kv_cache'] = decoder_size

        padding_size = 0.0
        if decoder_state.padding_cache is not None:
            for cache in decoder_state.padding_cache.layer_caches:
                if cache.cached_input is not None:
                    padding_size += _get_tensor_size_mb(cache.cached_input)
            sizes['padding_cache'] = padding_size
            decoder_size += padding_size

    total = gen_size + decoder_size
    breakdown = ", ".join(f"{k}={v:.2f}MB" for k, v in sizes.items())
    print(f"[CACHE] Seed {seed} size: {total:.2f}MB ({breakdown})", file=sys.stderr)


def _clone_decoder_state(decoder_state):
    """Deep clone a StreamingDecoderState to avoid mutating cached state."""
    if decoder_state is None:
        return None

    from dia2.audio.codec import StreamingDecoderState

    # Clone kv_cache (tuple of tensors)
    new_kv_cache = None
    if decoder_state.kv_cache is not None:
        # kv_cache is typically a tuple of tuples of tensors
        new_kv_cache = tuple(
            tuple(t.clone() if t is not None else None for t in layer_cache)
            for layer_cache in decoder_state.kv_cache
        )

    # Clone padding_cache (MimiDecoderPaddingCache with layer_caches)
    new_padding_cache = None
    if decoder_state.padding_cache is not None:
        import copy
        # Shallow copy the object, then clone the tensor data
        new_padding_cache = copy.copy(decoder_state.padding_cache)
        new_padding_cache.layer_caches = []
        for cache in decoder_state.padding_cache.layer_caches:
            new_cache = copy.copy(cache)
            if cache.cached_input is not None:
                new_cache.cached_input = cache.cached_input.clone()
            new_padding_cache.layer_caches.append(new_cache)

    return StreamingDecoderState(kv_cache=new_kv_cache, padding_cache=new_padding_cache)


def emit_event(event: dict):
    """Emit a JSON event to stdout."""
    try:
        print(json.dumps(event), flush=True)
    except BrokenPipeError:
        # Rust side closed the pipe - exit gracefully
        sys.exit(0)


def emit_status(message: str, progress: float):
    emit_event({
        "type": "status",
        "message": message,
        "progress": progress,
    })


def emit_audio(data: bytes, chunk_index: int, timestamp_ms: float, duration_ms: float):
    emit_event({
        "type": "audio_chunk",
        "data": base64.b64encode(data).decode('utf-8'),
        "chunk_index": chunk_index,
        "timestamp_ms": timestamp_ms,
        "duration_ms": duration_ms,
    })


def emit_complete(total_chunks: int, total_duration_ms: float):
    emit_event({
        "type": "complete",
        "total_chunks": total_chunks,
        "total_duration_ms": total_duration_ms,
    })


def emit_error(error: str):
    emit_event({
        "type": "error",
        "error": error,
    })


def load_model(model_size: str):
    """Load or reload the model if needed."""
    global _model, _device, _model_size

    import torch
    import os
    import time as time_module
    from dia2 import Dia2

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"

    # Check if model needs to be loaded
    if _model is not None and _model_size == model_size and _device == device:
        emit_status("Model already loaded", 0.2)
        runtime = _model._ensure_runtime()
        max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0
        print(f"[INFO] Model max_delay: {max_delay} frames (chunk_size must be > {max_delay})", file=sys.stderr)
        return _model

    load_start = time_module.time()
    emit_status(f"Loading Dia2-{model_size.upper()} on {device}...", 0.1)

    # Get model repo
    model_repo = f"nari-labs/Dia2-{model_size.upper()}"

    # Disable ALL progress bars to avoid corrupting JSON protocol on stdout
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["ACCELERATE_DISABLE_RICH"] = "1"

    # Also suppress tqdm globally
    import tqdm
    original_init = tqdm.tqdm.__init__
    def patched_init(self, *args, **kwargs):
        kwargs['file'] = sys.stderr
        kwargs['disable'] = True
        return original_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = patched_init

    # Load model with timing
    print(f"[TIMING] Starting Dia2.from_repo...", file=sys.stderr)
    t0 = time_module.time()
    model = Dia2.from_repo(model_repo, device=device, dtype=dtype)
    print(f"[TIMING] Dia2.from_repo: {time_module.time() - t0:.2f}s", file=sys.stderr)

    # Force runtime initialization to measure it separately
    print(f"[TIMING] Starting _ensure_runtime...", file=sys.stderr)
    t0 = time_module.time()
    runtime = model._ensure_runtime()
    print(f"[TIMING] _ensure_runtime (includes Mimi): {time_module.time() - t0:.2f}s", file=sys.stderr)

    print(f"[TIMING] Total load time: {time_module.time() - load_start:.2f}s", file=sys.stderr)

    # Store globally
    _model = model
    _device = device
    _model_size = model_size

    # Log max_delay for debugging/optimization
    runtime = model._ensure_runtime()
    max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0
    print(f"[INFO] Model loaded. max_delay: {max_delay} frames (chunk_size must be > {max_delay})", file=sys.stderr)

    emit_status("Model loaded", 0.2)
    return model


def process_request(request: dict):
    """Process a single TTS request with seed caching for fast repeated generations."""
    import tempfile
    import os
    import time as time_module
    import uuid

    # Generate unique request ID to sync with Rust side
    # This allows Rust to discard stale events from previous requests
    request_id = str(uuid.uuid4())
    emit_event({"type": "request_start", "request_id": request_id})

    from dia2 import (
        GenerationConfig,
        SamplingConfig,
        StreamingConfig,
        AudioChunkEvent,
        StatusEvent,
        CompleteEvent,
        ErrorEvent,
        normalize_script,
    )
    from dia2.runtime.generator import (
        build_initial_state,
        run_seed_warmup,
        run_streaming_generation_loop,
        decode_audio_streaming,
    )
    from dia2.audio.grid import undelay_frames
    from dia2.runtime.voice_clone import build_prefix_plan
    from dia2.generation import PrefixConfig, merge_generation_config
    from dia2.runtime.script_parser import parse_script
    from dia2.runtime.logger import RuntimeLogger

    text = request.get("text", "")
    model_size = request.get("model_size", "2b")
    config_overrides = request.get("config", {}) or {}

    if not text:
        emit_error("Text input is required")
        return

    # Handle voice cloning - base64 audio data for speakers
    prefix_speaker_1 = None
    prefix_speaker_2 = None
    temp_files = []

    try:
        # Handle voice cloning audio
        speaker_1_audio = config_overrides.get("speaker_1_audio")
        speaker_2_audio = config_overrides.get("speaker_2_audio")

        if speaker_1_audio:
            audio_bytes = base64.b64decode(speaker_1_audio)
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, audio_bytes)
            os.close(fd)
            prefix_speaker_1 = path
            temp_files.append(path)
            print(f"[INFO] Voice cloning: Speaker 1 audio saved to {path} ({len(audio_bytes)} bytes)", file=sys.stderr)

        if speaker_2_audio:
            audio_bytes = base64.b64decode(speaker_2_audio)
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.write(fd, audio_bytes)
            os.close(fd)
            prefix_speaker_2 = path
            temp_files.append(path)
            print(f"[INFO] Voice cloning: Speaker 2 audio saved to {path} ({len(audio_bytes)} bytes)", file=sys.stderr)

        # Load model
        model = load_model(model_size)
        runtime = model._ensure_runtime()

        # Handle random seed
        seed = config_overrides.get("seed")
        if seed is None or seed == "":
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = int(seed)

        # Set seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"[INFO] Using seed: {seed}", file=sys.stderr)

        # Emit the seed so frontend can display it
        emit_event({"type": "seed", "seed": seed})

        # Build generation config
        gen_config = GenerationConfig(
            text=SamplingConfig(
                temperature=config_overrides.get("text_temperature", 0.6),
                top_k=config_overrides.get("text_top_k", 50),
            ),
            audio=SamplingConfig(
                temperature=config_overrides.get("audio_temperature", 0.8),
                top_k=config_overrides.get("audio_top_k", 50),
            ),
            cfg_scale=config_overrides.get("cfg_scale", 2.0),
            cfg_filter_k=config_overrides.get("cfg_filter_k", 50),
            initial_padding=19,  # Must be >= max_delay (18) for caching
            use_cuda_graph=True,
            use_torch_compile=True,  # Enable torch.compile for faster inference
        )

        # Streaming config
        chunk_size = config_overrides.get("chunk_size_frames", 1)
        min_chunk = config_overrides.get("min_chunk_frames", 1)
        streaming_config = StreamingConfig(
            chunk_size_frames=chunk_size,
            min_chunk_frames=min_chunk,
            emit_status_every=config_overrides.get("emit_status_every", 20),
        )

        generation_start = time_module.time()

        # Check seed cache (only for non-voice-cloning requests)
        cached_state = None
        use_cache = not (prefix_speaker_1 or prefix_speaker_2)  # Don't cache voice cloning

        if use_cache:
            cached_state = _cache_get(seed)

        # Parse user text into entries (needed for both cache hit and miss)
        text_normalized = normalize_script(text)
        user_entries = parse_script([text_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate)

        if cached_state is not None:
            # FAST PATH: Restore from cache
            print(f"[CACHE] HIT - Restoring state for seed {seed}", file=sys.stderr)
            emit_status("Using cached voice...", 0.1)
            gen_state, state, rng_state, cuda_rng_state, warmup_steps, cached_decoder_state = cached_state

            # Clone all states so cache remains pristine
            gen_state = gen_state.clone()
            state = state.clone()
            decoder_state = _clone_decoder_state(cached_decoder_state)

            # Restore RNG state to match post-warmup state
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

            # Append user entries to the state (warmup entries already consumed)
            state.entries.extend(user_entries)
            state.end_step = None  # Reset so generation continues

            start_step = warmup_steps
        else:
            # SLOW PATH: Build initial state and run warmup
            print(f"[CACHE] MISS - Running warmup for seed {seed}", file=sys.stderr)
            emit_status("Warming up voice...", 0.1)

            # Build prefix plan for voice cloning
            prefix_config = None
            if prefix_speaker_1 or prefix_speaker_2:
                prefix_config = PrefixConfig(
                    speaker_1=prefix_speaker_1,
                    speaker_2=prefix_speaker_2,
                    include_audio=False,
                )
            prefix_plan = build_prefix_plan(runtime, prefix_config)

            # Build initial generation state
            gen_state = build_initial_state(runtime, prefix=prefix_plan)

            # Create state with warmup phrase entries
            # This exercises many phonemes while conditioning the voice.
            # S2 interjection creates a natural sentence boundary before user text.
            WARMUP_PHRASE = "[S1] Hello! This is a streaming TTS demo."
            warmup_entries = parse_script([WARMUP_PHRASE], runtime.tokenizer, runtime.constants, runtime.frame_rate)

            # Set initial_padding to 2 (original Dia default) for proper padding structure:
            # - 2 frames forced padding
            # - then warmup phrase processing
            # - total steps >= max_delay+1 to cover codec delay
            runtime.machine.initial_padding = 2
            state = runtime.machine.new_state(warmup_entries)
            runtime.machine.initial_padding = 0  # Reset to avoid side effects

            # Get max_delay for minimum warmup steps (codec alignment requirement)
            max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0

            # Run warmup until entries are consumed, minimum max_delay+1 steps
            gen_state, warmup_steps = run_seed_warmup(
                runtime,
                gen_state,
                gen_config,
                min_steps=max_delay + 1,  # Must be > max_delay for codec
                warmup_state=state,  # Use the SAME state that will continue for user text
            )
            start_step = warmup_steps

            # Build decoder state by decoding warmup audio (discard the audio, keep the state)
            # This primes the Mimi decoder's convolutional padding cache
            warmup_codes = gen_state.audio_buf[0, :, :warmup_steps + 1]  # (codebooks, steps)
            aligned_warmup = undelay_frames(
                warmup_codes,
                runtime.audio_delays,
                runtime.constants.audio_pad,
            ).unsqueeze(0)  # (1, codebooks, aligned_frames)
            _, decoder_state = decode_audio_streaming(runtime, aligned_warmup, None)
            print(f"[CACHE] Built decoder state from {aligned_warmup.shape[-1]} aligned warmup frames", file=sys.stderr)

            # Cache BOTH gen_state AND state (with warmup entries consumed)
            if use_cache:
                rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                _cache_put(seed, (gen_state.clone(), state.clone(), rng_state, cuda_rng_state, warmup_steps, decoder_state))
                _log_cache_entry_size(seed, gen_state, state, decoder_state)

            # Append user entries to the same state (warmup entries now consumed)
            state.entries.extend(user_entries)
            state.end_step = None  # Reset so generation continues

        warmup_time = time_module.time() - generation_start
        print(f"[TIMING] Warmup/restore took {warmup_time*1000:.0f}ms", file=sys.stderr)

        emit_status("Generating audio...", 0.2)

        # Run streaming generation
        logger = RuntimeLogger(enabled=False)
        event_count = 0
        audio_chunk_count = 0
        first_audio_time = None

        for event in run_streaming_generation_loop(
            runtime,
            state=state,
            generation=gen_state,
            config=gen_config,
            streaming_config=streaming_config,
            start_step=start_step,
            logger=logger,
            decoder_state=decoder_state,
        ):
            event_count += 1
            event_type = type(event).__name__

            if isinstance(event, AudioChunkEvent):
                audio_chunk_count += 1
                now = time_module.time()
                elapsed = now - generation_start

                if first_audio_time is None:
                    first_audio_time = elapsed
                    print(f"[TIMING] First audio chunk at {first_audio_time*1000:.0f}ms", file=sys.stderr)

                emit_audio(event.audio_data, event.chunk_index, event.timestamp_ms, event.duration_ms)

            elif isinstance(event, StatusEvent):
                emit_status(event.message, event.progress)

            elif isinstance(event, CompleteEvent):
                elapsed = time_module.time() - generation_start
                print(f"[TIMING] Complete: {event.total_chunks} chunks, {event.total_duration_ms:.0f}ms audio, total_time={elapsed*1000:.0f}ms", file=sys.stderr)
                emit_complete(event.total_chunks, event.total_duration_ms)

            elif isinstance(event, ErrorEvent):
                print(f"  Error: {event.error}", file=sys.stderr)
                emit_error(event.error)

        elapsed = time_module.time() - generation_start
        print(f"[TIMING] Generation finished. Events: {event_count}, Chunks: {audio_chunk_count}, Time: {elapsed*1000:.0f}ms", file=sys.stderr)

    except Exception as e:
        import traceback
        emit_error(f"Generation error: {e}")
        print(traceback.format_exc(), file=sys.stderr)

    finally:
        import os
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def main():
    """Main loop - process requests from stdin."""
    # Note: Don't emit events before receiving a request - it confuses the protocol
    print("TTS Bridge starting...", file=sys.stderr)

    # Handle SIGTERM gracefully
    def handle_signal(signum, frame):
        print("Received shutdown signal", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Import heavy modules upfront (logged to stderr only)
    try:
        import torch
        from dia2 import Dia2
        print("TTS Bridge modules imported", file=sys.stderr)
    except ImportError as e:
        emit_error(f"Failed to import required modules: {e}")
        return

    # Pre-load the model at startup (not on first request)
    # This ensures the model is ready when the first request arrives
    print("[STARTUP] Pre-loading model...", file=sys.stderr)
    try:
        load_model("2b")  # Default model size
        print("[STARTUP] Model pre-loaded and ready", file=sys.stderr)
    except Exception as e:
        import traceback
        print(f"[STARTUP] Warning: Failed to pre-load model: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        # Don't exit - model will be loaded on first request as fallback

    print("TTS Bridge ready", file=sys.stderr)

    # Main request loop
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                # EOF - stdin closed
                break

            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                emit_error(f"Invalid JSON: {e}")
                continue

            process_request(request)

        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            emit_error(f"Unexpected error: {e}")
            print(traceback.format_exc(), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        emit_error(f"Fatal error: {e}")
        print(traceback.format_exc(), file=sys.stderr)
