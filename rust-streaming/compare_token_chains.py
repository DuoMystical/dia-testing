#!/usr/bin/env python3
"""
Compare token chains between:
1. Full text parsed upfront (baseline)
2. Warmup phrase + extend with user text (current system)

This will help identify any differences in the token sequences.
"""

import sys
import os
import json
import random
import torch

# Add dia2-src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dia2-src'))

from dia2 import Dia2, GenerationConfig, SamplingConfig, normalize_script
from dia2.runtime.generator import build_initial_state, run_seed_warmup
from dia2.runtime.script_parser import parse_script
from dia2.runtime.state_machine import StateMachine, TokenIds


def log_entries(label: str, entries: list):
    """Log entries for debugging."""
    print(f"\n[{label}] Entries ({len(entries)} total):")
    for i, entry in enumerate(entries):
        print(f"  [{i}] text='{entry.text}', tokens={entry.tokens}, padding={entry.padding}")


def run_generation_with_logging(
    runtime,
    gen_state,
    gen_config,
    state,
    max_steps: int,
    label: str,
) -> list:
    """
    Run generation and log every step's token processing.
    Returns list of (step, main_token, consumed_text, state_info) tuples.
    """
    from dia2.runtime.generator import sample_text_token, sample_audio_tokens

    log = []

    print(f"\n{'='*60}")
    print(f"[{label}] Starting generation, max_steps={max_steps}")
    print(f"[{label}] Initial entries: {len(state.entries)}")
    print(f"[{label}] Initial padding_budget: {state.padding_budget}")
    print(f"[{label}] Initial forced_padding: {state.forced_padding}")
    print(f"{'='*60}")

    for t in range(max_steps):
        # Get state before processing
        entries_before = len(state.entries)
        pending_before = list(state.pending_tokens)
        forced_pad_before = state.forced_padding
        pad_budget_before = state.padding_budget
        transcript_before = len(state.transcript)

        # Sample text token
        text_logits = gen_state.decode.text_logits
        text_token = sample_text_token(text_logits, gen_config.text)

        # Process through state machine
        main_token, second_token, consumed = runtime.machine.process(
            t, state, text_token.item(), is_forced=(t < 2)  # First 2 steps forced
        )

        # Check what was consumed
        consumed_text = None
        if len(state.transcript) > transcript_before:
            consumed_text = state.transcript[-1][0]

        # Log this step
        step_info = {
            'step': t,
            'text_token': text_token.item(),
            'main_token': main_token,
            'consumed': consumed,
            'consumed_text': consumed_text,
            'entries_before': entries_before,
            'entries_after': len(state.entries),
            'pending_before': pending_before,
            'pending_after': list(state.pending_tokens),
            'forced_pad_before': forced_pad_before,
            'forced_pad_after': state.forced_padding,
            'pad_budget_before': pad_budget_before,
            'pad_budget_after': state.padding_budget,
            'end_step': state.end_step,
        }
        log.append(step_info)

        # Print key events
        token_name = 'PAD' if main_token == runtime.constants.pad else (
            'NEW_WORD' if main_token == runtime.constants.new_word else f'tok({main_token})'
        )
        if consumed_text:
            print(f"[{label}] t={t}: {token_name} -> CONSUMED '{consumed_text}' (entries: {entries_before}->{len(state.entries)}, pad_budget: {pad_budget_before}->{state.padding_budget})")
        elif t < 10 or consumed or state.end_step is not None:
            print(f"[{label}] t={t}: {token_name} (entries: {entries_before}, pad_budget: {pad_budget_before}, forced_pad: {forced_pad_before})")

        # Update generation state (simplified - just update step tokens)
        gen_state.step_tokens[0, 0, t] = main_token

        # Sample audio tokens (simplified)
        # In real code this would run depformer, but we just care about the text token chain

        # Check for end
        if state.end_step is not None and t >= state.end_step + 18:  # max_delay
            print(f"[{label}] t={t}: END (end_step={state.end_step})")
            break

    return log


def main():
    # Test text
    USER_TEXT = "Welcome to Banana Burgers! How can I help you today?"
    WARMUP_PHRASE = "[S1] Hello! This is a streaming TTS demo."
    FULL_TEXT = f"{WARMUP_PHRASE} {USER_TEXT}"

    # Use fixed seed for reproducibility
    SEED = 12345

    print("="*70)
    print("TOKEN CHAIN COMPARISON TEST")
    print("="*70)
    print(f"User text: {USER_TEXT}")
    print(f"Warmup phrase: {WARMUP_PHRASE}")
    print(f"Full text: {FULL_TEXT}")
    print(f"Seed: {SEED}")
    print("="*70)

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Dia2.from_repo("nari-labs/Dia2-2B", device=device, dtype="bfloat16" if device == "cuda" else "float32")
    runtime = model._ensure_runtime()

    gen_config = GenerationConfig(
        text=SamplingConfig(temperature=0.6, top_k=50),
        audio=SamplingConfig(temperature=0.8, top_k=50),
        cfg_scale=2.0,
        initial_padding=2,
    )

    max_delay = max(runtime.audio_delays) if runtime.audio_delays else 0
    print(f"max_delay: {max_delay}")

    # ========================================
    # APPROACH 1: Full text upfront (baseline)
    # ========================================
    print("\n" + "="*70)
    print("APPROACH 1: Full text parsed upfront (baseline)")
    print("="*70)

    # Set seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Parse full text
    full_text_normalized = normalize_script(FULL_TEXT)
    full_entries = parse_script([full_text_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate)
    log_entries("BASELINE", full_entries)

    # Build initial state
    runtime.machine.initial_padding = 2
    baseline_state = runtime.machine.new_state(full_entries)
    runtime.machine.initial_padding = 0

    # Build generation state
    baseline_gen_state = build_initial_state(runtime)

    # Run generation
    baseline_log = run_generation_with_logging(
        runtime, baseline_gen_state, gen_config, baseline_state,
        max_steps=100, label="BASELINE"
    )

    # ========================================
    # APPROACH 2: Warmup + Extend (current system)
    # ========================================
    print("\n" + "="*70)
    print("APPROACH 2: Warmup phrase + extend with user text")
    print("="*70)

    # Reset seed to same value
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Parse warmup phrase
    warmup_entries = parse_script([WARMUP_PHRASE], runtime.tokenizer, runtime.constants, runtime.frame_rate)
    log_entries("WARMUP", warmup_entries)

    # Parse user text (with initial_speaker_idx=0 to continue from S1)
    user_text_normalized = normalize_script(USER_TEXT)
    user_entries = parse_script([user_text_normalized], runtime.tokenizer, runtime.constants, runtime.frame_rate, initial_speaker_idx=0)
    log_entries("USER", user_entries)

    # Build initial state with warmup entries
    runtime.machine.initial_padding = 2
    extend_state = runtime.machine.new_state(warmup_entries)
    runtime.machine.initial_padding = 0

    # Build generation state
    extend_gen_state = build_initial_state(runtime)

    # Run warmup
    print("\n--- Running warmup phase ---")
    extend_gen_state, warmup_steps = run_seed_warmup(
        runtime,
        extend_gen_state,
        gen_config,
        min_steps=max_delay + 1,
        warmup_state=extend_state,
    )
    print(f"Warmup completed at step {warmup_steps}")
    print(f"State after warmup:")
    print(f"  end_step: {extend_state.end_step}")
    print(f"  padding_budget: {extend_state.padding_budget}")
    print(f"  forced_padding: {extend_state.forced_padding}")
    print(f"  pending_tokens: {list(extend_state.pending_tokens)}")
    print(f"  transcript: {extend_state.transcript}")

    # Extend with user entries (mimicking tts_bridge.py logic)
    print("\n--- Extending with user entries ---")
    print(f"Before extend: padding_budget={extend_state.padding_budget}, forced_padding={extend_state.forced_padding}")

    extend_state.end_step = None  # Allow generation to continue
    # Apply the fix: ensure minimum gap
    if extend_state.padding_budget < 2:
        extend_state.forced_padding = max(extend_state.forced_padding, 2)
        print(f"Applied minimum gap fix: forced_padding set to {extend_state.forced_padding}")
    extend_state.entries.extend(user_entries)

    print(f"After extend: padding_budget={extend_state.padding_budget}, forced_padding={extend_state.forced_padding}")
    print(f"Entries: {len(extend_state.entries)}")

    # Continue generation from warmup_steps
    print("\n--- Continuing generation after warmup ---")
    # Note: We can't easily continue the exact same generation state,
    # so we'll just log what the entries look like

    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON: Entry sequences")
    print("="*70)

    print("\nBASELINE entries (full text upfront):")
    for i, entry in enumerate(full_entries):
        print(f"  [{i}] '{entry.text}' -> tokens={entry.tokens}")

    print("\nWARMUP entries:")
    for i, entry in enumerate(warmup_entries):
        print(f"  [{i}] '{entry.text}' -> tokens={entry.tokens}")

    print("\nUSER entries (with initial_speaker_idx=0):")
    for i, entry in enumerate(user_entries):
        print(f"  [{i}] '{entry.text}' -> tokens={entry.tokens}")

    print("\nCOMBINED (warmup + user):")
    combined = list(warmup_entries) + list(user_entries)
    for i, entry in enumerate(combined):
        print(f"  [{i}] '{entry.text}' -> tokens={entry.tokens}")

    # Check if they match
    print("\n" + "="*70)
    print("TOKEN SEQUENCE COMPARISON")
    print("="*70)

    baseline_tokens = []
    for entry in full_entries:
        baseline_tokens.extend(entry.tokens)

    combined_tokens = []
    for entry in combined:
        combined_tokens.extend(entry.tokens)

    print(f"\nBaseline token count: {len(baseline_tokens)}")
    print(f"Combined token count: {len(combined_tokens)}")

    if baseline_tokens == combined_tokens:
        print("\n*** TOKENS MATCH EXACTLY ***")
    else:
        print("\n*** TOKENS DIFFER ***")
        print("\nBaseline tokens:", baseline_tokens)
        print("\nCombined tokens:", combined_tokens)

        # Find first difference
        for i in range(max(len(baseline_tokens), len(combined_tokens))):
            bt = baseline_tokens[i] if i < len(baseline_tokens) else None
            ct = combined_tokens[i] if i < len(combined_tokens) else None
            if bt != ct:
                print(f"\nFirst difference at index {i}:")
                print(f"  Baseline: {bt}")
                print(f"  Combined: {ct}")
                break

    # Export full logs as JSON
    output = {
        'user_text': USER_TEXT,
        'warmup_phrase': WARMUP_PHRASE,
        'full_text': FULL_TEXT,
        'seed': SEED,
        'baseline_entries': [{'text': e.text, 'tokens': e.tokens, 'padding': e.padding} for e in full_entries],
        'warmup_entries': [{'text': e.text, 'tokens': e.tokens, 'padding': e.padding} for e in warmup_entries],
        'user_entries': [{'text': e.text, 'tokens': e.tokens, 'padding': e.padding} for e in user_entries],
        'baseline_tokens': baseline_tokens,
        'combined_tokens': combined_tokens,
        'tokens_match': baseline_tokens == combined_tokens,
    }

    with open('token_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull comparison exported to token_comparison.json")


if __name__ == "__main__":
    main()
