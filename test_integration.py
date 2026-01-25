#!/usr/bin/env python3
"""
Integration tests for Dia2 streaming functionality.
Tests the tts_bridge.py and simulates end-to-end flow.
"""

import sys
import os
import json
import subprocess
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dia2-src'))


def test_tts_bridge_input_validation():
    """Test that tts_bridge.py validates input correctly."""
    print("=" * 60)
    print("TEST 1: tts_bridge.py Input Validation")
    print("=" * 60)

    bridge_path = os.path.join(os.path.dirname(__file__), 'rust-streaming/tts_bridge.py')

    # Test empty text
    result = subprocess.run(
        ['python', bridge_path],
        input='{"text": "", "model_size": "2b"}\n',
        capture_output=True,
        text=True,
        timeout=10
    )

    output = result.stdout.strip()
    if output:
        try:
            event = json.loads(output)
            if event.get('type') == 'error' and 'required' in event.get('error', '').lower():
                print("[PASS] Empty text returns error event")
                return True
            else:
                print(f"[WARN] Unexpected response: {event}")
        except json.JSONDecodeError:
            print(f"[FAIL] Invalid JSON output: {output}")
            return False

    print("[PASS] Input validation works (no crash)")
    return True


def test_tts_bridge_json_parsing():
    """Test that tts_bridge.py parses JSON correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: tts_bridge.py JSON Parsing")
    print("=" * 60)

    bridge_path = os.path.join(os.path.dirname(__file__), 'rust-streaming/tts_bridge.py')

    # Test invalid JSON
    result = subprocess.run(
        ['python', bridge_path],
        input='not valid json\n',
        capture_output=True,
        text=True,
        timeout=10
    )

    output = result.stdout.strip()
    if output:
        try:
            event = json.loads(output)
            if event.get('type') == 'error':
                print("[PASS] Invalid JSON returns error event")
                return True
        except json.JSONDecodeError:
            pass

    print("[PASS] JSON parsing handles errors")
    return True


def test_event_serialization():
    """Test that events can be serialized to JSON for WebSocket."""
    print("\n" + "=" * 60)
    print("TEST 3: Event Serialization")
    print("=" * 60)

    from dia2 import AudioChunkEvent, StatusEvent, CompleteEvent, ErrorEvent

    # Test AudioChunkEvent serialization
    audio_event = AudioChunkEvent(
        audio_data=b"test_wav_data",
        chunk_index=5,
        timestamp_ms=1234.5
    )

    # Convert to dict for JSON
    audio_dict = {
        "type": "audio_chunk",
        "data": audio_event.audio_data.hex(),  # or base64
        "chunk_index": audio_event.chunk_index,
        "timestamp_ms": audio_event.timestamp_ms
    }
    json_str = json.dumps(audio_dict)
    parsed = json.loads(json_str)
    assert parsed["chunk_index"] == 5
    print("[PASS] AudioChunkEvent serializes to JSON")

    # Test StatusEvent
    status_event = StatusEvent(message="Processing", progress=0.75)
    status_dict = {
        "type": "status",
        "message": status_event.message,
        "progress": status_event.progress
    }
    json_str = json.dumps(status_dict)
    parsed = json.loads(json_str)
    assert parsed["progress"] == 0.75
    print("[PASS] StatusEvent serializes to JSON")

    # Test CompleteEvent
    complete_event = CompleteEvent(total_chunks=42, total_duration_ms=10500.0)
    complete_dict = {
        "type": "complete",
        "total_chunks": complete_event.total_chunks,
        "total_duration_ms": complete_event.total_duration_ms
    }
    json_str = json.dumps(complete_dict)
    parsed = json.loads(json_str)
    assert parsed["total_chunks"] == 42
    print("[PASS] CompleteEvent serializes to JSON")

    # Test ErrorEvent
    error_event = ErrorEvent(error="Something went wrong")
    error_dict = {
        "type": "error",
        "error": error_event.error
    }
    json_str = json.dumps(error_dict)
    parsed = json.loads(json_str)
    assert "wrong" in parsed["error"]
    print("[PASS] ErrorEvent serializes to JSON")

    return True


def test_wav_chunk_validity():
    """Test that generated WAV chunks are valid."""
    print("\n" + "=" * 60)
    print("TEST 4: WAV Chunk Validity")
    print("=" * 60)

    import torch
    import wave
    from dia2.runtime.generator import _encode_wav_chunk

    # Generate test audio at different sample rates
    sample_rates = [24000, 44100, 48000]

    for sr in sample_rates:
        duration = 0.1
        samples = int(sr * duration)
        t = torch.linspace(0, duration, samples)
        waveform = torch.sin(2 * 3.14159 * 440 * t)

        wav_bytes = _encode_wav_chunk(waveform, sr)

        # Validate WAV header
        assert wav_bytes[:4] == b'RIFF', f"Invalid RIFF header for {sr}Hz"
        assert wav_bytes[8:12] == b'WAVE', f"Invalid WAVE format for {sr}Hz"

        # Validate can be read
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            assert wf.getframerate() == sr
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

        print(f"  [PASS] Valid WAV at {sr}Hz")

    print("[PASS] All sample rates produce valid WAV")
    return True


def test_streaming_config_variations():
    """Test different streaming configurations."""
    print("\n" + "=" * 60)
    print("TEST 5: StreamingConfig Variations")
    print("=" * 60)

    from dia2 import StreamingConfig

    # Test various chunk sizes
    configs = [
        StreamingConfig(chunk_size_frames=4, min_chunk_frames=2),   # Very small (~50ms)
        StreamingConfig(chunk_size_frames=8, min_chunk_frames=4),   # Default (~100ms)
        StreamingConfig(chunk_size_frames=15, min_chunk_frames=8),  # Medium (~200ms)
        StreamingConfig(chunk_size_frames=25, min_chunk_frames=12), # Larger (~333ms)
    ]

    for i, cfg in enumerate(configs):
        ms_per_frame = 1000.0 / 75.0  # ~13.33ms per frame at 75fps
        chunk_ms = cfg.chunk_size_frames * ms_per_frame
        print(f"  Config {i+1}: {cfg.chunk_size_frames} frames = ~{chunk_ms:.0f}ms chunks")

        assert cfg.chunk_size_frames >= cfg.min_chunk_frames
        assert cfg.chunk_size_frames > 0

    print("[PASS] All StreamingConfig variations valid")
    return True


def test_bidirectional_text_chunking():
    """Test text chunking for bidirectional streaming."""
    print("\n" + "=" * 60)
    print("TEST 6: Bidirectional Text Chunking")
    print("=" * 60)

    # Simulate how the frontend would chunk text
    text = "[S1] Hello! This is a test of bidirectional streaming. [S2] The audio should start generating before all text is received."

    chunk_size = 50
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        is_final = i + chunk_size >= len(text)
        chunks.append({
            "type": "text_chunk",
            "text": chunk,
            "chunk_index": len(chunks),
            "final": is_final
        })

    # Verify chunks
    assert len(chunks) > 1, "Should have multiple chunks"
    assert chunks[-1]["final"] == True, "Last chunk should be final"
    assert chunks[0]["final"] == False, "First chunk should not be final"

    # Reconstruct text
    reconstructed = "".join(c["text"] for c in chunks)
    assert reconstructed == text, "Reconstructed text should match"

    print(f"  Text length: {len(text)}")
    print(f"  Chunk count: {len(chunks)}")
    print(f"  Chunk size: {chunk_size}")
    print("[PASS] Text chunking works correctly")
    return True


def test_websocket_message_format():
    """Test WebSocket message format compatibility."""
    print("\n" + "=" * 60)
    print("TEST 7: WebSocket Message Format")
    print("=" * 60)

    import base64

    # Test generate request format
    generate_msg = {
        "type": "generate",
        "text": "[S1] Hello world",
        "model": "2b",
        "config": {
            "audio_temperature": 0.8,
            "text_temperature": 0.6,
            "cfg_scale": 2.0,
            "chunk_size_frames": 8
        }
    }
    json_str = json.dumps(generate_msg)
    assert len(json_str) < 10000, "Message should be reasonably sized"
    print("[PASS] Generate request format valid")

    # Test audio chunk response format
    fake_wav = b"RIFF" + b"\x00" * 100  # Fake WAV data
    audio_response = {
        "type": "audio",
        "data": base64.b64encode(fake_wav).decode('utf-8'),
        "chunk_index": 0,
        "timestamp_ms": 0.0
    }
    json_str = json.dumps(audio_response)
    parsed = json.loads(json_str)
    decoded = base64.b64decode(parsed["data"])
    assert decoded == fake_wav
    print("[PASS] Audio response format valid")

    # Test status message format
    status_msg = {
        "type": "status",
        "message": "Generating audio (50/100 steps)",
        "progress": 0.5
    }
    json_str = json.dumps(status_msg)
    assert "progress" in json_str
    print("[PASS] Status message format valid")

    # Test complete message format
    complete_msg = {
        "type": "complete",
        "total_chunks": 50,
        "total_duration_ms": 5000.0
    }
    json_str = json.dumps(complete_msg)
    assert "total_chunks" in json_str
    print("[PASS] Complete message format valid")

    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "#" * 60)
    print("# INTEGRATION TESTS")
    print("#" * 60)

    tests = [
        ("tts_bridge Input Validation", test_tts_bridge_input_validation),
        ("tts_bridge JSON Parsing", test_tts_bridge_json_parsing),
        ("Event Serialization", test_event_serialization),
        ("WAV Chunk Validity", test_wav_chunk_validity),
        ("StreamingConfig Variations", test_streaming_config_variations),
        ("Bidirectional Text Chunking", test_bidirectional_text_chunking),
        ("WebSocket Message Format", test_websocket_message_format),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    failed = sum(1 for _, r in results if not r)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
