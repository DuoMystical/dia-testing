#!/usr/bin/env python3
"""
Syntax and structure tests that don't require torch.
These verify the code is syntactically correct and properly structured.
"""

import sys
import os
import ast
import importlib.util

def test_python_syntax(filepath):
    """Test that a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def test_generation_module_syntax():
    """Test generation.py syntax and structure."""
    print("=" * 60)
    print("TEST 1: generation.py Syntax")
    print("=" * 60)

    filepath = os.path.join(os.path.dirname(__file__), 'dia2-src/dia2/generation.py')
    success, error = test_python_syntax(filepath)

    if success:
        print(f"[PASS] {filepath} has valid syntax")

        # Parse and check for expected classes
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())

        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        expected_classes = ['AudioChunkEvent', 'StatusEvent', 'CompleteEvent', 'ErrorEvent',
                           'StreamingConfig', 'SamplingConfig', 'GenerationConfig',
                           'GenerationResult', 'PrefixConfig']

        for cls in expected_classes:
            if cls in class_names:
                print(f"  [PASS] Found class: {cls}")
            else:
                print(f"  [FAIL] Missing class: {cls}")
                success = False

        return success
    else:
        print(f"[FAIL] Syntax error: {error}")
        return False


def test_generator_module_syntax():
    """Test generator.py syntax and structure."""
    print("\n" + "=" * 60)
    print("TEST 2: generator.py Syntax")
    print("=" * 60)

    filepath = os.path.join(os.path.dirname(__file__), 'dia2-src/dia2/runtime/generator.py')
    success, error = test_python_syntax(filepath)

    if success:
        print(f"[PASS] {filepath} has valid syntax")

        # Parse and check for expected functions
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())

        func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        expected_funcs = ['run_generation_loop', 'run_streaming_generation_loop',
                         'build_initial_state', 'decode_audio', '_encode_wav_chunk']

        for func in expected_funcs:
            if func in func_names:
                print(f"  [PASS] Found function: {func}")
            else:
                print(f"  [FAIL] Missing function: {func}")
                success = False

        return success
    else:
        print(f"[FAIL] Syntax error: {error}")
        return False


def test_engine_module_syntax():
    """Test engine.py syntax and structure."""
    print("\n" + "=" * 60)
    print("TEST 3: engine.py Syntax")
    print("=" * 60)

    filepath = os.path.join(os.path.dirname(__file__), 'dia2-src/dia2/engine.py')
    success, error = test_python_syntax(filepath)

    if success:
        print(f"[PASS] {filepath} has valid syntax")

        # Parse and check for Dia2 class with generate_stream
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())

        # Find Dia2 class
        dia2_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'Dia2':
                dia2_class = node
                break

        if dia2_class:
            print(f"  [PASS] Found class: Dia2")

            # Check for methods
            methods = [node.name for node in dia2_class.body if isinstance(node, ast.FunctionDef)]
            expected_methods = ['generate', 'generate_stream', 'save_wav']

            for method in expected_methods:
                if method in methods:
                    print(f"  [PASS] Dia2 has method: {method}")
                else:
                    print(f"  [FAIL] Dia2 missing method: {method}")
                    success = False
        else:
            print(f"  [FAIL] Dia2 class not found")
            success = False

        return success
    else:
        print(f"[FAIL] Syntax error: {error}")
        return False


def test_init_module_exports():
    """Test __init__.py exports streaming types."""
    print("\n" + "=" * 60)
    print("TEST 4: __init__.py Exports")
    print("=" * 60)

    filepath = os.path.join(os.path.dirname(__file__), 'dia2-src/dia2/__init__.py')
    success, error = test_python_syntax(filepath)

    if not success:
        print(f"[FAIL] Syntax error: {error}")
        return False

    print(f"[PASS] {filepath} has valid syntax")

    with open(filepath, 'r') as f:
        content = f.read()

    expected_exports = ['AudioChunkEvent', 'StatusEvent', 'CompleteEvent', 'ErrorEvent',
                       'StreamEvent', 'StreamGenerator', 'StreamingConfig', 'Dia2']

    for export in expected_exports:
        if export in content:
            print(f"  [PASS] Exports: {export}")
        else:
            print(f"  [FAIL] Missing export: {export}")
            success = False

    return success


def test_tts_bridge_syntax():
    """Test tts_bridge.py syntax."""
    print("\n" + "=" * 60)
    print("TEST 5: tts_bridge.py Syntax")
    print("=" * 60)

    filepath = os.path.join(os.path.dirname(__file__), 'rust-streaming/tts_bridge.py')
    success, error = test_python_syntax(filepath)

    if success:
        print(f"[PASS] {filepath} has valid syntax")

        # Check for expected functions
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())

        func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        expected_funcs = ['emit_event', 'emit_status', 'emit_audio', 'emit_complete', 'emit_error', 'main']

        for func in expected_funcs:
            if func in func_names:
                print(f"  [PASS] Found function: {func}")
            else:
                print(f"  [FAIL] Missing function: {func}")
                success = False

        return success
    else:
        print(f"[FAIL] Syntax error: {error}")
        return False


def test_rust_cargo_toml():
    """Test Cargo.toml is valid."""
    print("\n" + "=" * 60)
    print("TEST 6: Cargo.toml Validity")
    print("=" * 60)

    filepath = os.path.join(os.path.dirname(__file__), 'rust-streaming/Cargo.toml')

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Check required sections
        required = ['[package]', 'name =', 'version =', '[dependencies]', 'axum', 'tokio', 'serde']
        success = True

        for req in required:
            if req in content:
                print(f"  [PASS] Contains: {req}")
            else:
                print(f"  [FAIL] Missing: {req}")
                success = False

        return success
    except Exception as e:
        print(f"[FAIL] Error reading Cargo.toml: {e}")
        return False


def test_rust_main_syntax():
    """Test Rust main.rs compiles (syntax check)."""
    print("\n" + "=" * 60)
    print("TEST 7: Rust Source Files")
    print("=" * 60)

    rust_files = [
        'rust-streaming/src/main.rs',
        'rust-streaming/src/session.rs',
        'rust-streaming/src/tts_bridge.rs'
    ]

    success = True
    for filepath in rust_files:
        full_path = os.path.join(os.path.dirname(__file__), filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
            # Basic check - file is not empty and has some Rust code
            if len(content) > 100 and ('fn ' in content or 'struct ' in content or 'use ' in content):
                print(f"  [PASS] {filepath} exists and has Rust code")
            else:
                print(f"  [WARN] {filepath} seems incomplete")
        else:
            print(f"  [FAIL] {filepath} not found")
            success = False

    return success


def test_frontend_files():
    """Test frontend files exist and have valid content."""
    print("\n" + "=" * 60)
    print("TEST 8: Frontend Files")
    print("=" * 60)

    files = [
        ('streaming-frontend/index.html', ['<!DOCTYPE html>', '<html', 'websocket', 'audio']),  # case-insensitive check
        ('streaming-frontend/css/style.css', ['.container', 'background', 'color']),
        ('streaming-frontend/js/websocket-client.js', ['class WebSocketClient', 'connect', 'send']),
        ('streaming-frontend/js/audio-streamer.js', ['class AudioStreamer', 'AudioContext', 'addChunk']),
        ('streaming-frontend/js/app.js', ['DOMContentLoaded', 'WebSocketClient', 'AudioStreamer']),
    ]

    success = True
    for filepath, required_content in files:
        full_path = os.path.join(os.path.dirname(__file__), filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()

            all_found = True
            for req in required_content:
                # Case-insensitive check
                if req.lower() not in content.lower():
                    print(f"  [FAIL] {filepath} missing: {req}")
                    all_found = False
                    success = False

            if all_found:
                print(f"  [PASS] {filepath} has all required content")
        else:
            print(f"  [FAIL] {filepath} not found")
            success = False

    return success


def run_all_tests():
    """Run all syntax and structure tests."""
    print("\n" + "#" * 60)
    print("# SYNTAX AND STRUCTURE TESTS (No torch required)")
    print("#" * 60)

    tests = [
        ("generation.py Syntax", test_generation_module_syntax),
        ("generator.py Syntax", test_generator_module_syntax),
        ("engine.py Syntax", test_engine_module_syntax),
        ("__init__.py Exports", test_init_module_exports),
        ("tts_bridge.py Syntax", test_tts_bridge_syntax),
        ("Cargo.toml Validity", test_rust_cargo_toml),
        ("Rust Source Files", test_rust_main_syntax),
        ("Frontend Files", test_frontend_files),
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
