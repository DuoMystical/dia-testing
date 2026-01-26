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


def emit_event(event: dict):
    """Emit a JSON event to stdout."""
    print(json.dumps(event), flush=True)


def emit_status(message: str, progress: float):
    emit_event({
        "type": "status",
        "message": message,
        "progress": progress,
    })


def emit_audio(data: bytes, chunk_index: int, timestamp_ms: float):
    emit_event({
        "type": "audio_chunk",
        "data": base64.b64encode(data).decode('utf-8'),
        "chunk_index": chunk_index,
        "timestamp_ms": timestamp_ms,
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
    from dia2 import Dia2

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"

    # Check if model needs to be loaded
    if _model is not None and _model_size == model_size and _device == device:
        emit_status("Model already loaded", 0.2)
        # Log max_delay for debugging/optimization
        runtime = _model._ensure_runtime()
        max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0
        print(f"[INFO] Model max_delay: {max_delay} frames (chunk_size must be > {max_delay})", file=sys.stderr)
        return _model

    emit_status(f"Loading Dia2-{model_size.upper()} on {device}...", 0.1)

    # Get model repo
    model_repo = f"nari-labs/Dia2-{model_size.upper()}"

    # Load model
    model = Dia2.from_repo(model_repo, device=device, dtype=dtype)

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
    )
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
            cfg_scale=config_overrides.get("cfg_scale", 6.0),
            cfg_filter_k=config_overrides.get("cfg_filter_k", 50),
            initial_padding=19,  # Must be >= max_delay (18) for caching
            use_cuda_graph=True,
            use_torch_compile=False,
        )

        # Streaming config
        chunk_size = config_overrides.get("chunk_size_frames", 1)
        min_chunk = config_overrides.get("min_chunk_frames", 1)
        streaming_config = StreamingConfig(
            chunk_size_frames=chunk_size,
            min_chunk_frames=min_chunk,
            emit_status_every=config_overrides.get("emit_status_every", 20),
        )

        print(f"[CONFIG] chunk_size_frames={chunk_size}, min_chunk_frames={min_chunk}", file=sys.stderr)

        generation_start = time_module.time()

        # Check seed cache (only for non-voice-cloning requests)
        cached_state = None
        use_cache = not (prefix_speaker_1 or prefix_speaker_2)  # Don't cache voice cloning

        if use_cache:
            cached_state = _cache_get(seed)

        if cached_state is not None:
            # FAST PATH: Restore from cache
            print(f"[CACHE] HIT - Restoring state for seed {seed}", file=sys.stderr)
            emit_status("Using cached voice...", 0.1)
            gen_state = cached_state.clone()
            start_step = gen_config.initial_padding
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

            # Build initial state
            gen_state = build_initial_state(runtime, prefix=prefix_plan)

            # Run warmup (seed-dependent, text-independent)
            gen_state = run_seed_warmup(
                runtime,
                gen_state,
                gen_config,
                num_steps=gen_config.initial_padding,
            )
            start_step = gen_config.initial_padding

            # Cache the warmed-up state (only if not voice cloning)
            if use_cache:
                _cache_put(seed, gen_state.clone())

        warmup_time = time_module.time() - generation_start
        print(f"[TIMING] Warmup/restore took {warmup_time*1000:.0f}ms", file=sys.stderr)

        emit_status("Generating audio...", 0.2)

        # Parse script and create state machine
        text_normalized = normalize_script(text)
        entries = parse_script([text_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate)

        # Set initial_padding to 0 since warmup is done
        runtime.machine.initial_padding = 0
        state = runtime.machine.new_state(entries)
        # Reset forced_padding since warmup already consumed it
        state.forced_padding = 0

        # Run streaming generation
        logger = RuntimeLogger(enabled=False)
        event_count = 0
        audio_chunk_count = 0
        first_audio_time = None

        print(f"Starting generation with text: {text[:50]}...", file=sys.stderr)

        for event in run_streaming_generation_loop(
            runtime,
            state=state,
            generation=gen_state,
            config=gen_config,
            streaming_config=streaming_config,
            start_step=start_step,
            logger=logger,
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

                audio_bytes = len(event.audio_data) - 44
                chunk_duration_ms = (audio_bytes / 48000) * 1000
                print(f"  AudioChunk #{audio_chunk_count}: {len(event.audio_data)} bytes ({chunk_duration_ms:.0f}ms audio), elapsed={elapsed*1000:.0f}ms", file=sys.stderr)
                emit_audio(event.audio_data, event.chunk_index, event.timestamp_ms)

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
        print("TTS Bridge ready", file=sys.stderr)
    except ImportError as e:
        emit_error(f"Failed to import required modules: {e}")
        return

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
