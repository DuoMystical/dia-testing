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

# Global model state
_model = None
_device = None
_model_size = None


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
    """Process a single TTS request."""
    import tempfile
    import os
    from dia2 import (
        GenerationConfig,
        SamplingConfig,
        StreamingConfig,
        AudioChunkEvent,
        StatusEvent,
        CompleteEvent,
        ErrorEvent,
    )

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
            # Decode base64 and write to temp file
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

        # Load model (will be fast if already loaded)
        model = load_model(model_size)

        emit_status("Starting generation...", 0.2)

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
            use_cuda_graph=True,
            use_torch_compile=False,
        )

        # Streaming config - chunk_size must be larger than max audio delay (~12-16 frames)
        # Using 32 frames (~0.4s at 75fps) to ensure chunks have actual audio after undelaying
        streaming_config = StreamingConfig(
            chunk_size_frames=config_overrides.get("chunk_size_frames", 32),
            min_chunk_frames=config_overrides.get("min_chunk_frames", 16),
            emit_status_every=config_overrides.get("emit_status_every", 5),
        )

        # Generate with streaming (include voice cloning if provided)
        event_count = 0
        audio_chunk_count = 0
        print(f"Starting generate_stream with text: {text[:50]}...", file=sys.stderr)
        if prefix_speaker_1 or prefix_speaker_2:
            print(f"  Voice cloning enabled: S1={prefix_speaker_1}, S2={prefix_speaker_2}", file=sys.stderr)

        for event in model.generate_stream(
            text,
            config=gen_config,
            streaming_config=streaming_config,
            prefix_speaker_1=prefix_speaker_1,
            prefix_speaker_2=prefix_speaker_2,
            verbose=False,  # Must be False - verbose=True outputs to stdout and corrupts JSON protocol
        ):
            event_count += 1
            event_type = type(event).__name__
            print(f"Received event #{event_count}: {event_type}", file=sys.stderr)

            if isinstance(event, AudioChunkEvent):
                audio_chunk_count += 1
                print(f"  AudioChunk #{audio_chunk_count}: {len(event.audio_data)} bytes, index={event.chunk_index}", file=sys.stderr)
                emit_audio(event.audio_data, event.chunk_index, event.timestamp_ms)
            elif isinstance(event, StatusEvent):
                print(f"  Status: {event.message}, progress={event.progress}", file=sys.stderr)
                emit_status(event.message, event.progress)
            elif isinstance(event, CompleteEvent):
                print(f"  Complete: {event.total_chunks} chunks, {event.total_duration_ms}ms", file=sys.stderr)
                emit_complete(event.total_chunks, event.total_duration_ms)
            elif isinstance(event, ErrorEvent):
                print(f"  Error: {event.error}", file=sys.stderr)
                emit_error(event.error)
            else:
                print(f"  Unknown event type: {event}", file=sys.stderr)

        print(f"Generation loop finished. Total events: {event_count}, Audio chunks: {audio_chunk_count}", file=sys.stderr)

    except Exception as e:
        import traceback
        emit_error(f"Generation error: {e}")
        print(traceback.format_exc(), file=sys.stderr)

    finally:
        # Clean up temp files for voice cloning
        import os
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
                print(f"[INFO] Cleaned up temp file: {temp_file}", file=sys.stderr)
            except Exception as cleanup_err:
                print(f"[WARN] Failed to clean up {temp_file}: {cleanup_err}", file=sys.stderr)


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
