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

    import copy
    from dia2.audio.codec import StreamingDecoderState

    # Clone kv_cache - it's a HuggingFace DynamicCache object, use deepcopy
    new_kv_cache = None
    if decoder_state.kv_cache is not None:
        new_kv_cache = copy.deepcopy(decoder_state.kv_cache)

    # Clone padding_cache (MimiDecoderPaddingCache with layer_caches)
    new_padding_cache = None
    if decoder_state.padding_cache is not None:
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
        run_generation_loop,
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
        debug_include_warmup = config_overrides.get("debug_include_warmup", False)
        debug_skip_warmup = config_overrides.get("debug_skip_warmup", False)
        if debug_skip_warmup:
            print(f"[DEBUG] debug_skip_warmup=True: running in baseline mode (no warmup)", file=sys.stderr)
        streaming_config = StreamingConfig(
            chunk_size_frames=chunk_size,
            min_chunk_frames=min_chunk,
            emit_status_every=config_overrides.get("emit_status_every", 20),
            debug_include_warmup=debug_include_warmup,
        )
        if debug_include_warmup:
            print(f"[DEBUG] debug_include_warmup=True: will output all audio including warmup", file=sys.stderr)

        generation_start = time_module.time()

        # Check seed cache (only for non-voice-cloning requests)
        cached_state = None
        use_cache = not (prefix_speaker_1 or prefix_speaker_2)  # Don't cache voice cloning

        if use_cache:
            cached_state = _cache_get(seed)

        # Parse user text into entries (needed for both cache hit and miss)
        # Warmup phrase is "[S1] Hello! This is a streaming TTS demo." so last speaker is S1 (index 0)
        # Pass initial_speaker_idx=0 to continue seamlessly without re-inserting [S1] token
        text_normalized = normalize_script(text)
        user_entries = parse_script([text_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate, initial_speaker_idx=0)

        # Debug: log user entries
        print(f"[DEBUG ENTRIES] User entries ({len(user_entries)} total):", file=sys.stderr)
        for i, entry in enumerate(user_entries[:5]):  # First 5 entries
            print(f"[DEBUG ENTRIES]   [{i}] tokens={entry.tokens}, text='{entry.text}', padding={entry.padding}", file=sys.stderr)

        # Debug: compare entry parsing between baseline (full text) and warmup+user
        debug_compare_entries = config_overrides.get("debug_compare_entries", False)
        if debug_compare_entries:
            WARMUP_PHRASE = "[S1] Hello! This is a streaming TTS demo."
            FULL_TEXT = f"{WARMUP_PHRASE} {text}"

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[COMPARE] Entry Parsing Comparison", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"[COMPARE] Warmup: {WARMUP_PHRASE}", file=sys.stderr)
            print(f"[COMPARE] User:   {text}", file=sys.stderr)
            print(f"[COMPARE] Full:   {FULL_TEXT}", file=sys.stderr)

            # Parse full text (baseline)
            full_normalized = normalize_script(FULL_TEXT)
            baseline_entries = parse_script([full_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate)

            # Parse warmup separately
            warmup_entries_for_compare = parse_script([WARMUP_PHRASE], runtime.tokenizer, runtime.constants, runtime.frame_rate)

            # Combined = warmup + user (user already parsed with initial_speaker_idx=0)
            combined_entries = list(warmup_entries_for_compare) + list(user_entries)

            print(f"\n[COMPARE] Baseline entries ({len(baseline_entries)}):", file=sys.stderr)
            for i, entry in enumerate(baseline_entries):
                print(f"[COMPARE]   [{i:2d}] text='{entry.text}' tokens={entry.tokens} padding={entry.padding}", file=sys.stderr)

            print(f"\n[COMPARE] Warmup entries ({len(warmup_entries_for_compare)}):", file=sys.stderr)
            for i, entry in enumerate(warmup_entries_for_compare):
                print(f"[COMPARE]   [{i:2d}] text='{entry.text}' tokens={entry.tokens} padding={entry.padding}", file=sys.stderr)

            print(f"\n[COMPARE] User entries ({len(user_entries)}):", file=sys.stderr)
            for i, entry in enumerate(user_entries):
                print(f"[COMPARE]   [{i:2d}] text='{entry.text}' tokens={entry.tokens} padding={entry.padding}", file=sys.stderr)

            print(f"\n[COMPARE] Combined entries ({len(combined_entries)}):", file=sys.stderr)
            for i, entry in enumerate(combined_entries):
                print(f"[COMPARE]   [{i:2d}] text='{entry.text}' tokens={entry.tokens} padding={entry.padding}", file=sys.stderr)

            # Extract token sequences
            baseline_tokens = []
            for entry in baseline_entries:
                baseline_tokens.extend(entry.tokens)

            combined_tokens = []
            for entry in combined_entries:
                combined_tokens.extend(entry.tokens)

            print(f"\n[COMPARE] Token sequences:", file=sys.stderr)
            print(f"[COMPARE]   Baseline ({len(baseline_tokens)}): {baseline_tokens}", file=sys.stderr)
            print(f"[COMPARE]   Combined ({len(combined_tokens)}): {combined_tokens}", file=sys.stderr)

            if baseline_tokens == combined_tokens:
                print(f"\n[COMPARE] *** TOKENS MATCH ***", file=sys.stderr)
            else:
                print(f"\n[COMPARE] *** TOKENS DIFFER ***", file=sys.stderr)
                for i in range(max(len(baseline_tokens), len(combined_tokens))):
                    bt = baseline_tokens[i] if i < len(baseline_tokens) else None
                    ct = combined_tokens[i] if i < len(combined_tokens) else None
                    if bt != ct:
                        print(f"[COMPARE]   First diff at index {i}: baseline={bt}, combined={ct}", file=sys.stderr)
                        break
            print(f"{'='*60}\n", file=sys.stderr)

        # Debug: compare AUDIO tokens between baseline and warmup+extend
        # This runs baseline generation first, saves audio tokens, then lets warmup+extend run
        debug_compare_audio_tokens = config_overrides.get("debug_compare_audio_tokens", False)
        baseline_audio_buf = None
        baseline_end_step = None

        if debug_compare_audio_tokens:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[AUDIO COMPARE] Running baseline generation for comparison", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            # Save RNG state so we can restore it for warmup+extend
            saved_rng_state = torch.get_rng_state()
            saved_cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

            # Reset RNG to same seed for baseline
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Build initial state for baseline
            baseline_gen_state = build_initial_state(runtime, prefix=None)

            # Parse FULL text (warmup + user) for baseline
            WARMUP_PHRASE = "[S1] Hello! This is a streaming TTS demo."
            FULL_TEXT = f"{WARMUP_PHRASE} {text}"
            full_normalized = normalize_script(FULL_TEXT)
            baseline_entries = parse_script([full_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate)

            # Create baseline state with full text (use default initial_padding)
            baseline_state = runtime.machine.new_state(baseline_entries)

            # Create baseline gen config - this is the CORRECT config that produces good audio
            baseline_gen_config = GenerationConfig(
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
                initial_padding=19,  # Baseline uses 19 (>= max_delay for proper codec alignment)
                use_cuda_graph=True,
                use_torch_compile=True,
            )

            # Run baseline generation (non-streaming)
            print(f"[AUDIO COMPARE] Running baseline with {len(baseline_entries)} entries...", file=sys.stderr)
            import time as time_module
            baseline_start = time_module.time()

            first_word_frame, baseline_tokens_result = run_generation_loop(
                runtime,
                state=baseline_state,
                generation=baseline_gen_state,
                config=baseline_gen_config,
                start_step=0,
            )

            baseline_elapsed = time_module.time() - baseline_start
            print(f"[AUDIO COMPARE] Baseline completed in {baseline_elapsed:.2f}s", file=sys.stderr)
            print(f"[AUDIO COMPARE] Baseline end_step: {baseline_state.end_step}", file=sys.stderr)
            print(f"[AUDIO COMPARE] Baseline audio_buf shape: {baseline_gen_state.audio_buf.shape}", file=sys.stderr)
            print(f"[AUDIO COMPARE] Baseline word timing: {baseline_state.transcript}", file=sys.stderr)

            # Save baseline audio buffer for later comparison
            baseline_audio_buf = baseline_gen_state.audio_buf.clone()
            baseline_end_step = baseline_state.end_step
            baseline_transcript = list(baseline_state.transcript)  # Save for comparison

            # Restore RNG state for warmup+extend run
            torch.set_rng_state(saved_rng_state)
            if saved_cuda_rng_state is not None:
                torch.cuda.set_rng_state(saved_cuda_rng_state)

            # Re-seed for warmup+extend (same seed as baseline started with)
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"[AUDIO COMPARE] Now running warmup+extend for comparison...", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)

        # BASELINE MODE: Skip warmup entirely if debug flag is set
        if debug_skip_warmup:
            print(f"[BASELINE MODE] Running without warmup...", file=sys.stderr)
            emit_status("Generating audio (baseline mode)...", 0.1)

            # Build prefix plan for voice cloning (same as normal path)
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

            # Parse user text directly (no warmup phrase)
            # Use fresh parse without initial_speaker_idx to auto-insert [S1] if needed
            text_normalized = normalize_script(text)
            baseline_user_entries = parse_script([text_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate)

            # Create state with user entries only
            state = runtime.machine.new_state(baseline_user_entries)

            # Start from step 0, no warmup
            start_step = 0
            decoder_state = None  # No warmup audio, no decoder state

            print(f"[BASELINE MODE] Starting generation from step 0 with {len(baseline_user_entries)} entries", file=sys.stderr)

        elif cached_state is not None:
            # FAST PATH: Restore from cache
            print(f"[CACHE] HIT - Restoring state for seed {seed}", file=sys.stderr)
            emit_status("Using cached voice...", 0.1)
            gen_state, state, rng_state, cuda_rng_state, warmup_steps, cached_decoder_state = cached_state

            # Clone all states so cache remains pristine
            gen_state = gen_state.clone()
            state = state.clone()
            decoder_state = _clone_decoder_state(cached_decoder_state)

            # Debug: log restored state info
            max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0
            print(f"[DEBUG HIT] warmup_steps={warmup_steps}, max_delay={max_delay}", file=sys.stderr)
            print(f"[DEBUG HIT] gen_state.audio_buf.shape={gen_state.audio_buf.shape}", file=sys.stderr)
            if decoder_state is not None:
                print(f"[DEBUG HIT] decoder_state.kv_cache type={type(decoder_state.kv_cache).__name__}", file=sys.stderr)
                if decoder_state.kv_cache is not None and hasattr(decoder_state.kv_cache, 'get_seq_length'):
                    print(f"[DEBUG HIT] decoder_state.kv_cache.get_seq_length()={decoder_state.kv_cache.get_seq_length()}", file=sys.stderr)

            # Restore RNG state to match post-warmup state
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)

            # Extend with user entries - mimic normal algorithm where entries are in sequence
            # DON'T reset padding_budget/forced_padding/pending_tokens - let natural state carry through
            # After warmup flush, these should already be 0/empty (that's what triggered end_step)
            # Only reset end_step to allow generation to continue
            print(f"[DEBUG EXTEND] State before extend (cache HIT):", file=sys.stderr)
            print(f"[DEBUG EXTEND]   padding_budget: {state.padding_budget}", file=sys.stderr)
            print(f"[DEBUG EXTEND]   forced_padding: {state.forced_padding}", file=sys.stderr)
            print(f"[DEBUG EXTEND]   pending_tokens: {list(state.pending_tokens)}", file=sys.stderr)
            print(f"[DEBUG EXTEND]   end_step: {state.end_step}", file=sys.stderr)

            state.end_step = None  # Allow generation to continue
            state.entries.extend(user_entries)

            # Debug: log state after adding user entries
            print(f"[DEBUG STATE] After cache HIT + user entries:", file=sys.stderr)
            print(f"[DEBUG STATE]   entries count: {len(state.entries)}", file=sys.stderr)
            print(f"[DEBUG STATE]   padding_budget: {state.padding_budget}", file=sys.stderr)
            print(f"[DEBUG STATE]   forced_padding: {state.forced_padding}", file=sys.stderr)
            print(f"[DEBUG STATE]   pending_tokens: {list(state.pending_tokens)}", file=sys.stderr)
            print(f"[DEBUG STATE]   end_step: {state.end_step}", file=sys.stderr)
            # Show first few entries that will be processed
            print(f"[DEBUG STATE]   first 3 entries to process:", file=sys.stderr)
            for i, entry in enumerate(list(state.entries)[:3]):
                print(f"[DEBUG STATE]     [{i}] text='{entry.text}', tokens={entry.tokens}, padding={entry.padding}", file=sys.stderr)

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

            # Debug: log warmup entries
            print(f"[DEBUG ENTRIES] Warmup entries ({len(warmup_entries)} total):", file=sys.stderr)
            for i, entry in enumerate(warmup_entries):
                print(f"[DEBUG ENTRIES]   [{i}] tokens={entry.tokens}, text='{entry.text}', padding={entry.padding}", file=sys.stderr)

            # Create state with default initial_padding (matches baseline behavior)
            # Baseline doesn't set runtime.machine.initial_padding, so neither should warmup
            state = runtime.machine.new_state(warmup_entries)

            # Get max_delay for minimum warmup steps (codec alignment requirement)
            max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0

            # Run warmup until phrase is consumed, minimum max_delay+1 steps
            gen_state, warmup_steps = run_seed_warmup(
                runtime,
                gen_state,
                gen_config,
                min_steps=max_delay + 1,  # Must be > max_delay for codec
                warmup_state=state,  # Use the SAME state that will continue for user text
            )
            start_step = warmup_steps

            # Debug: log state RIGHT after warmup (before user entries added)
            print(f"[DEBUG STATE] State RIGHT after warmup (before user entries):", file=sys.stderr)
            print(f"[DEBUG STATE]   warmup_steps: {warmup_steps}", file=sys.stderr)
            print(f"[DEBUG STATE]   end_step: {state.end_step}", file=sys.stderr)
            print(f"[DEBUG STATE]   entries count: {len(state.entries)}", file=sys.stderr)
            print(f"[DEBUG STATE]   padding_budget: {state.padding_budget}", file=sys.stderr)
            print(f"[DEBUG STATE]   forced_padding: {state.forced_padding}", file=sys.stderr)
            print(f"[DEBUG STATE]   pending_tokens: {list(state.pending_tokens)}", file=sys.stderr)

            # Build decoder state by decoding warmup audio (discard the audio, keep the state)
            # This primes the Mimi decoder's convolutional padding cache
            warmup_codes = gen_state.audio_buf[0, :, :warmup_steps + 1]  # (codebooks, steps)
            print(f"[DEBUG WARMUP] warmup_steps={warmup_steps}, raw_frames={warmup_codes.shape[-1]}, max_delay={max_delay}", file=sys.stderr)
            aligned_warmup = undelay_frames(
                warmup_codes,
                runtime.audio_delays,
                runtime.constants.audio_pad,
            ).unsqueeze(0)  # (1, codebooks, aligned_frames)
            print(f"[DEBUG WARMUP] aligned_warmup.shape={aligned_warmup.shape}, aligned_frames={aligned_warmup.shape[-1]}", file=sys.stderr)
            warmup_audio, decoder_state = decode_audio_streaming(runtime, aligned_warmup, None)
            warmup_audio_samples = warmup_audio.numel() if warmup_audio is not None else 0
            warmup_audio_ms = (warmup_audio_samples / 24000) * 1000 if warmup_audio_samples > 0 else 0
            print(f"[DEBUG WARMUP] Decoded warmup: {warmup_audio_samples} samples = {warmup_audio_ms:.1f}ms audio", file=sys.stderr)
            print(f"[DEBUG WARMUP] decoder_state.kv_cache type={type(decoder_state.kv_cache).__name__}", file=sys.stderr)
            if decoder_state.kv_cache is not None and hasattr(decoder_state.kv_cache, 'get_seq_length'):
                print(f"[DEBUG WARMUP] decoder_state.kv_cache.get_seq_length()={decoder_state.kv_cache.get_seq_length()}", file=sys.stderr)
            print(f"[CACHE] Built decoder state from {aligned_warmup.shape[-1]} aligned warmup frames", file=sys.stderr)

            # Cache BOTH gen_state AND state (with warmup entries consumed)
            if use_cache:
                print(f"[DEBUG CACHE] Caching state for seed {seed}:", file=sys.stderr)
                print(f"[DEBUG CACHE]   warmup_steps: {warmup_steps}", file=sys.stderr)
                print(f"[DEBUG CACHE]   state.end_step: {state.end_step}", file=sys.stderr)
                print(f"[DEBUG CACHE]   state.padding_budget: {state.padding_budget}", file=sys.stderr)
                print(f"[DEBUG CACHE]   state.forced_padding: {state.forced_padding}", file=sys.stderr)
                print(f"[DEBUG CACHE]   state.pending_tokens: {list(state.pending_tokens)}", file=sys.stderr)
                print(f"[DEBUG CACHE]   state.entries: {len(state.entries)}", file=sys.stderr)
                print(f"[DEBUG CACHE]   state.transcript: {state.transcript}", file=sys.stderr)
                rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                _cache_put(seed, (gen_state.clone(), state.clone(), rng_state, cuda_rng_state, warmup_steps, decoder_state))
                _log_cache_entry_size(seed, gen_state, state, decoder_state)

            # Extend with user entries - mimic normal algorithm where entries are in sequence
            # DON'T reset padding_budget/forced_padding/pending_tokens - let natural state carry through
            # After warmup flush, these should already be 0/empty (that's what triggered end_step)
            # Only reset end_step to allow generation to continue
            print(f"[DEBUG EXTEND] State before extend (cache MISS):", file=sys.stderr)
            print(f"[DEBUG EXTEND]   padding_budget: {state.padding_budget}", file=sys.stderr)
            print(f"[DEBUG EXTEND]   forced_padding: {state.forced_padding}", file=sys.stderr)
            print(f"[DEBUG EXTEND]   pending_tokens: {list(state.pending_tokens)}", file=sys.stderr)
            print(f"[DEBUG EXTEND]   end_step: {state.end_step}", file=sys.stderr)

            state.end_step = None  # Allow generation to continue
            state.entries.extend(user_entries)

            # Debug: log state after adding user entries
            print(f"[DEBUG STATE] After cache MISS + user entries:", file=sys.stderr)
            print(f"[DEBUG STATE]   entries count: {len(state.entries)}", file=sys.stderr)
            print(f"[DEBUG STATE]   padding_budget: {state.padding_budget}", file=sys.stderr)
            print(f"[DEBUG STATE]   forced_padding: {state.forced_padding}", file=sys.stderr)
            print(f"[DEBUG STATE]   pending_tokens: {list(state.pending_tokens)}", file=sys.stderr)
            print(f"[DEBUG STATE]   end_step: {state.end_step}", file=sys.stderr)
            # Show first few entries that will be processed
            print(f"[DEBUG STATE]   first 3 entries to process:", file=sys.stderr)
            for i, entry in enumerate(list(state.entries)[:3]):
                print(f"[DEBUG STATE]     [{i}] text='{entry.text}', tokens={entry.tokens}, padding={entry.padding}", file=sys.stderr)

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

        # Compare audio tokens if baseline was run
        if baseline_audio_buf is not None:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[AUDIO COMPARE] Comparing audio tokens", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            # Get warmup+extend audio buffer
            warmup_extend_audio_buf = gen_state.audio_buf

            print(f"[AUDIO COMPARE] Baseline audio_buf shape: {baseline_audio_buf.shape}", file=sys.stderr)
            print(f"[AUDIO COMPARE] Warmup+extend audio_buf shape: {warmup_extend_audio_buf.shape}", file=sys.stderr)
            print(f"[AUDIO COMPARE] Baseline end_step: {baseline_end_step}", file=sys.stderr)
            print(f"[AUDIO COMPARE] Warmup+extend end_step: {state.end_step}", file=sys.stderr)

            # Compare CB0 tokens (most important codebook) at key positions
            max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0

            # Compare from position 1 to min of both buffer lengths
            compare_end = min(baseline_audio_buf.shape[-1], warmup_extend_audio_buf.shape[-1])

            # CB0 is at index 0
            baseline_cb0 = baseline_audio_buf[0, 0, :compare_end].cpu()
            warmup_cb0 = warmup_extend_audio_buf[0, 0, :compare_end].cpu()

            # Find first difference
            diff_mask = baseline_cb0 != warmup_cb0
            num_diffs = diff_mask.sum().item()

            if num_diffs == 0:
                print(f"[AUDIO COMPARE] *** CB0 TOKENS MATCH (positions 0-{compare_end-1}) ***", file=sys.stderr)
            else:
                first_diff_idx = diff_mask.nonzero()[0].item()
                print(f"[AUDIO COMPARE] *** CB0 TOKENS DIFFER ***", file=sys.stderr)
                print(f"[AUDIO COMPARE]   Total differences: {num_diffs} out of {compare_end} positions", file=sys.stderr)
                print(f"[AUDIO COMPARE]   First difference at position {first_diff_idx}", file=sys.stderr)

                # Show tokens around the first difference
                start = max(0, first_diff_idx - 3)
                end = min(compare_end, first_diff_idx + 5)
                print(f"[AUDIO COMPARE]   Baseline CB0[{start}:{end}]: {baseline_cb0[start:end].tolist()}", file=sys.stderr)
                print(f"[AUDIO COMPARE]   Warmup+extend CB0[{start}:{end}]: {warmup_cb0[start:end].tolist()}", file=sys.stderr)

                # Show where in generation this corresponds to
                # Position N was written at step N-1
                print(f"[AUDIO COMPARE]   Position {first_diff_idx} was written at step {first_diff_idx - 1}", file=sys.stderr)

                # Check if difference is in warmup region or user region
                # Warmup region: positions 1 to warmup_steps (approximately)
                # User region: positions warmup_steps+1 onwards
                if first_diff_idx <= start_step:
                    print(f"[AUDIO COMPARE]   This is in the WARMUP region (before step {start_step})", file=sys.stderr)
                else:
                    print(f"[AUDIO COMPARE]   This is in the USER region (at/after step {start_step})", file=sys.stderr)

            # Also compare all codebooks at a few key positions
            print(f"\n[AUDIO COMPARE] Comparing all codebooks at key positions:", file=sys.stderr)
            key_positions = [start_step, start_step + 1, start_step + 5, start_step + 10]
            for pos in key_positions:
                if pos < compare_end:
                    baseline_all = baseline_audio_buf[0, :, pos].cpu().tolist()
                    warmup_all = warmup_extend_audio_buf[0, :, pos].cpu().tolist()
                    match = baseline_all == warmup_all
                    status = "MATCH" if match else "DIFFER"
                    print(f"[AUDIO COMPARE]   Position {pos}: {status}", file=sys.stderr)
                    if not match:
                        print(f"[AUDIO COMPARE]     Baseline: {baseline_all[:4]}...", file=sys.stderr)
                        print(f"[AUDIO COMPARE]     Warmup+extend: {warmup_all[:4]}...", file=sys.stderr)

            print(f"{'='*60}\n", file=sys.stderr)

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
