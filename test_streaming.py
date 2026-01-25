#!/usr/bin/env python3
"""
Comprehensive test suite for Dia2 streaming functionality.
Tests both streaming and non-streaming APIs.
"""

import sys
import os

# Add dia2-src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dia2-src'))

def test_imports():
    """Test that all streaming types can be imported."""
    print("=" * 60)
    print("TEST 1: Import Tests")
    print("=" * 60)

    try:
        from dia2 import (
            Dia2,
            GenerationConfig,
            GenerationResult,
            SamplingConfig,
            PrefixConfig,
            # Streaming types
            AudioChunkEvent,
            StatusEvent,
            CompleteEvent,
            ErrorEvent,
            StreamEvent,
            StreamGenerator,
            StreamingConfig,
        )
        print("[PASS] All streaming types imported successfully")

        # Verify types are correct
        assert AudioChunkEvent is not None
        assert StatusEvent is not None
        assert CompleteEvent is not None
        assert ErrorEvent is not None
        assert StreamingConfig is not None
        print("[PASS] All types are valid")

        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def test_streaming_config():
    """Test StreamingConfig creation and defaults."""
    print("\n" + "=" * 60)
    print("TEST 2: StreamingConfig Tests")
    print("=" * 60)

    try:
        from dia2 import StreamingConfig

        # Test default config
        config = StreamingConfig()
        assert config.chunk_size_frames == 8, f"Expected 8, got {config.chunk_size_frames}"
        assert config.min_chunk_frames == 4, f"Expected 4, got {config.min_chunk_frames}"
        assert config.emit_status_every == 5, f"Expected 5, got {config.emit_status_every}"
        print(f"[PASS] Default config: chunk_size={config.chunk_size_frames}, min_chunk={config.min_chunk_frames}")

        # Test custom config
        custom = StreamingConfig(
            chunk_size_frames=12,
            min_chunk_frames=6,
            emit_status_every=10
        )
        assert custom.chunk_size_frames == 12
        assert custom.min_chunk_frames == 6
        assert custom.emit_status_every == 10
        print(f"[PASS] Custom config: chunk_size={custom.chunk_size_frames}")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_event_types():
    """Test that event types can be instantiated."""
    print("\n" + "=" * 60)
    print("TEST 3: Event Type Tests")
    print("=" * 60)

    try:
        from dia2 import AudioChunkEvent, StatusEvent, CompleteEvent, ErrorEvent

        # Test AudioChunkEvent
        audio_event = AudioChunkEvent(
            audio_data=b"fake_wav_data",
            chunk_index=0,
            timestamp_ms=0.0
        )
        assert audio_event.audio_data == b"fake_wav_data"
        assert audio_event.chunk_index == 0
        assert audio_event.timestamp_ms == 0.0
        print("[PASS] AudioChunkEvent created successfully")

        # Test StatusEvent
        status_event = StatusEvent(
            message="Testing",
            progress=0.5
        )
        assert status_event.message == "Testing"
        assert status_event.progress == 0.5
        print("[PASS] StatusEvent created successfully")

        # Test CompleteEvent
        complete_event = CompleteEvent(
            total_chunks=10,
            total_duration_ms=5000.0
        )
        assert complete_event.total_chunks == 10
        assert complete_event.total_duration_ms == 5000.0
        print("[PASS] CompleteEvent created successfully")

        # Test ErrorEvent
        error_event = ErrorEvent(error="Test error")
        assert error_event.error == "Test error"
        print("[PASS] ErrorEvent created successfully")

        # Test frozen (immutable)
        try:
            audio_event.chunk_index = 5
            print("[FAIL] Event should be frozen/immutable")
            return False
        except Exception:
            print("[PASS] Events are frozen (immutable) as expected")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_config_with_streaming():
    """Test that GenerationConfig works with streaming."""
    print("\n" + "=" * 60)
    print("TEST 4: GenerationConfig Tests")
    print("=" * 60)

    try:
        from dia2 import GenerationConfig, SamplingConfig

        # Test default config
        config = GenerationConfig()
        assert config.cfg_scale == 2.0
        assert config.text.temperature == 0.6
        assert config.audio.temperature == 0.8
        print(f"[PASS] Default GenerationConfig: cfg_scale={config.cfg_scale}")

        # Test custom config
        custom = GenerationConfig(
            text=SamplingConfig(temperature=0.7, top_k=40),
            audio=SamplingConfig(temperature=0.9, top_k=60),
            cfg_scale=3.0,
            use_cuda_graph=False  # Important for streaming
        )
        assert custom.cfg_scale == 3.0
        assert custom.text.temperature == 0.7
        assert custom.use_cuda_graph == False
        print(f"[PASS] Custom GenerationConfig: cfg_scale={custom.cfg_scale}")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dia2_class_has_generate_stream():
    """Test that Dia2 class has generate_stream method."""
    print("\n" + "=" * 60)
    print("TEST 5: Dia2.generate_stream Method Tests")
    print("=" * 60)

    try:
        from dia2 import Dia2

        # Check method exists
        assert hasattr(Dia2, 'generate_stream'), "Dia2 should have generate_stream method"
        print("[PASS] Dia2 has generate_stream method")

        # Check method signature (via inspection)
        import inspect
        sig = inspect.signature(Dia2.generate_stream)
        params = list(sig.parameters.keys())

        expected_params = ['self', 'script', 'config', 'streaming_config',
                          'prefix_speaker_1', 'prefix_speaker_2', 'include_prefix', 'verbose']
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"
        print(f"[PASS] generate_stream has correct signature: {params}")

        # Check original generate still exists
        assert hasattr(Dia2, 'generate'), "Dia2 should still have generate method"
        print("[PASS] Dia2 still has original generate method")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generator_module():
    """Test the generator module has streaming functions."""
    print("\n" + "=" * 60)
    print("TEST 6: Generator Module Tests")
    print("=" * 60)

    try:
        from dia2.runtime.generator import (
            run_generation_loop,
            run_streaming_generation_loop,
            build_initial_state,
            decode_audio,
            warmup_with_prefix,
            GenerationState,
        )
        print("[PASS] All generator functions imported")

        # Check run_streaming_generation_loop signature
        import inspect
        sig = inspect.signature(run_streaming_generation_loop)
        params = list(sig.parameters.keys())

        assert 'streaming_config' in params, "Should have streaming_config parameter"
        print(f"[PASS] run_streaming_generation_loop has streaming_config parameter")

        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wav_encoding():
    """Test that WAV encoding helper works."""
    print("\n" + "=" * 60)
    print("TEST 7: WAV Encoding Tests")
    print("=" * 60)

    try:
        # Import the internal function
        from dia2.runtime.generator import _encode_wav_chunk
        import torch

        # Create a simple test waveform
        sample_rate = 24000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)

        # Generate a simple sine wave
        t = torch.linspace(0, duration, samples)
        waveform = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz tone

        # Encode to WAV
        wav_bytes = _encode_wav_chunk(waveform, sample_rate)

        # Check it's valid WAV (starts with RIFF header)
        assert wav_bytes[:4] == b'RIFF', "Should start with RIFF header"
        assert wav_bytes[8:12] == b'WAVE', "Should have WAVE format"
        print(f"[PASS] WAV encoding works: {len(wav_bytes)} bytes for {duration}s audio")

        # Verify we can decode it
        import io
        import wave

        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            assert wf.getnchannels() == 1, "Should be mono"
            assert wf.getsampwidth() == 2, "Should be 16-bit"
            assert wf.getframerate() == sample_rate, f"Should be {sample_rate}Hz"
            frames = wf.getnframes()
            print(f"[PASS] WAV decoded: {frames} frames, {wf.getnchannels()} channels, {wf.getframerate()}Hz")

        return True
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "#" * 60)
    print("# DIA2 STREAMING FUNCTIONALITY TESTS")
    print("#" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("StreamingConfig Tests", test_streaming_config),
        ("Event Type Tests", test_event_types),
        ("GenerationConfig Tests", test_generation_config_with_streaming),
        ("Dia2.generate_stream Tests", test_dia2_class_has_generate_stream),
        ("Generator Module Tests", test_generator_module),
        ("WAV Encoding Tests", test_wav_encoding),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
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
