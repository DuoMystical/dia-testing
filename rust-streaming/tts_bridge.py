#!/usr/bin/env python3
"""
TTS Bridge for Dia2 Streaming Server

This script is called by the Rust WebSocket server via subprocess.
It reads a JSON request from stdin, generates audio using Dia2's streaming API,
and outputs JSON events to stdout.
"""

import json
import sys
import base64

# Note: dia2 is installed via pip in the Docker image, no need to add to path


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


def main():
    # Read request from stdin
    try:
        request_line = sys.stdin.readline()
        if not request_line:
            emit_error("No input received")
            return

        request = json.loads(request_line)
    except json.JSONDecodeError as e:
        emit_error(f"Invalid JSON input: {e}")
        return
    except Exception as e:
        emit_error(f"Error reading input: {e}")
        return

    text = request.get("text", "")
    model_size = request.get("model_size", "2b")
    config_overrides = request.get("config", {}) or {}

    if not text:
        emit_error("Text input is required")
        return

    emit_status("Loading Dia2 model...", 0.0)

    try:
        import torch
        from dia2 import (
            Dia2,
            GenerationConfig,
            SamplingConfig,
            StreamingConfig,
            AudioChunkEvent,
            StatusEvent,
            CompleteEvent,
            ErrorEvent,
        )

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16" if device == "cuda" else "float32"

        emit_status(f"Loading Dia2-{model_size.upper()} on {device}...", 0.1)

        # Get model repo
        model_repo = f"nari-labs/Dia2-{model_size.upper()}"

        # Load model
        model = Dia2.from_repo(model_repo, device=device, dtype=dtype)

        emit_status("Model loaded, starting generation...", 0.2)

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
            use_cuda_graph=False,  # Disabled for streaming
            use_torch_compile=config_overrides.get("use_torch_compile", False),
        )

        # Streaming config - small chunks for responsive streaming
        streaming_config = StreamingConfig(
            chunk_size_frames=config_overrides.get("chunk_size_frames", 8),
            min_chunk_frames=config_overrides.get("min_chunk_frames", 4),
            emit_status_every=config_overrides.get("emit_status_every", 5),
        )

        # Generate with streaming
        for event in model.generate_stream(
            text,
            config=gen_config,
            streaming_config=streaming_config,
            verbose=False,
        ):
            if isinstance(event, AudioChunkEvent):
                emit_audio(event.audio_data, event.chunk_index, event.timestamp_ms)
            elif isinstance(event, StatusEvent):
                emit_status(event.message, event.progress)
            elif isinstance(event, CompleteEvent):
                emit_complete(event.total_chunks, event.total_duration_ms)
            elif isinstance(event, ErrorEvent):
                emit_error(event.error)

    except ImportError as e:
        emit_error(f"Failed to import dia2: {e}. Make sure dia2 is installed.")
    except Exception as e:
        emit_error(f"Generation error: {e}")
        import traceback
        print(traceback.format_exc(), file=sys.stderr)


if __name__ == "__main__":
    main()
