#!/usr/bin/env python3
"""
Simple comparison of parsed entries between:
1. Full text parsed upfront (baseline)
2. Warmup phrase + user text parsed separately (with initial_speaker_idx=0)

This doesn't require loading the full model - just the tokenizer.
"""

import sys
import os

# Add dia2-src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dia2-src'))

from transformers import AutoTokenizer
from dia2.runtime.script_parser import parse_script
from dia2.runtime.state_machine import TokenIds
from dia2 import normalize_script


def main():
    # Test text
    USER_TEXT = "Welcome to Banana Burgers! How can I help you today?"
    WARMUP_PHRASE = "[S1] Hello! This is a streaming TTS demo."
    FULL_TEXT = f"{WARMUP_PHRASE} {USER_TEXT}"

    print("="*70)
    print("ENTRY PARSING COMPARISON")
    print("="*70)
    print(f"Warmup: {WARMUP_PHRASE}")
    print(f"User:   {USER_TEXT}")
    print(f"Full:   {FULL_TEXT}")
    print("="*70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("nari-labs/Dia2-2B")

    # Token IDs (from Dia2 constants)
    constants = TokenIds(
        card=2051,
        new_word=2,
        pad=3,
        bos=1,
        zero=0,
        spk1=49152,
        spk2=49153,
        audio_pad=2048,
        audio_bos=2049,
    )
    frame_rate = 12.5

    # ========================================
    # Parse full text (baseline)
    # ========================================
    print("\n" + "-"*50)
    print("BASELINE: Full text parsed upfront")
    print("-"*50)

    full_normalized = normalize_script(FULL_TEXT)
    print(f"Normalized: {full_normalized}")

    full_entries = parse_script([full_normalized], tokenizer, constants, frame_rate)

    print(f"\nEntries ({len(full_entries)}):")
    for i, entry in enumerate(full_entries):
        print(f"  [{i:2d}] text='{entry.text:20s}' tokens={entry.tokens}, padding={entry.padding}")

    # ========================================
    # Parse warmup + user separately
    # ========================================
    print("\n" + "-"*50)
    print("WARMUP + USER: Parsed separately")
    print("-"*50)

    # Warmup (no initial_speaker_idx - starts fresh)
    warmup_entries = parse_script([WARMUP_PHRASE], tokenizer, constants, frame_rate)
    print(f"\nWarmup entries ({len(warmup_entries)}):")
    for i, entry in enumerate(warmup_entries):
        print(f"  [{i:2d}] text='{entry.text:20s}' tokens={entry.tokens}, padding={entry.padding}")

    # User text (with initial_speaker_idx=0 to continue from S1)
    user_normalized = normalize_script(USER_TEXT)
    print(f"\nUser text normalized: {user_normalized}")

    user_entries = parse_script([user_normalized], tokenizer, constants, frame_rate, initial_speaker_idx=0)
    print(f"\nUser entries ({len(user_entries)}):")
    for i, entry in enumerate(user_entries):
        print(f"  [{i:2d}] text='{entry.text:20s}' tokens={entry.tokens}, padding={entry.padding}")

    # ========================================
    # Compare
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    # Combine warmup + user
    combined_entries = list(warmup_entries) + list(user_entries)

    print(f"\nBaseline entry count: {len(full_entries)}")
    print(f"Combined entry count: {len(combined_entries)}")

    # Extract all tokens
    baseline_tokens = []
    for entry in full_entries:
        baseline_tokens.extend(entry.tokens)

    combined_tokens = []
    for entry in combined_entries:
        combined_tokens.extend(entry.tokens)

    print(f"\nBaseline token count: {len(baseline_tokens)}")
    print(f"Combined token count: {len(combined_tokens)}")

    # Compare entry by entry
    print("\n" + "-"*50)
    print("Entry-by-entry comparison:")
    print("-"*50)

    max_entries = max(len(full_entries), len(combined_entries))
    all_match = True

    for i in range(max_entries):
        baseline = full_entries[i] if i < len(full_entries) else None
        combined = combined_entries[i] if i < len(combined_entries) else None

        if baseline is None:
            print(f"[{i:2d}] BASELINE: (missing)")
            print(f"     COMBINED: text='{combined.text}' tokens={combined.tokens}")
            all_match = False
        elif combined is None:
            print(f"[{i:2d}] BASELINE: text='{baseline.text}' tokens={baseline.tokens}")
            print(f"     COMBINED: (missing)")
            all_match = False
        elif baseline.tokens != combined.tokens or baseline.text != combined.text:
            print(f"[{i:2d}] BASELINE: text='{baseline.text}' tokens={baseline.tokens}")
            print(f"     COMBINED: text='{combined.text}' tokens={combined.tokens}")
            print(f"     *** MISMATCH ***")
            all_match = False
        else:
            print(f"[{i:2d}] MATCH: text='{baseline.text}' tokens={baseline.tokens}")

    print("\n" + "="*70)
    if all_match:
        print("*** ALL ENTRIES MATCH ***")
    else:
        print("*** ENTRIES DIFFER ***")
    print("="*70)

    # Token sequence comparison
    print("\nFull token sequences:")
    print(f"Baseline: {baseline_tokens}")
    print(f"Combined: {combined_tokens}")

    if baseline_tokens == combined_tokens:
        print("\n*** TOKEN SEQUENCES MATCH ***")
    else:
        print("\n*** TOKEN SEQUENCES DIFFER ***")
        for i in range(max(len(baseline_tokens), len(combined_tokens))):
            bt = baseline_tokens[i] if i < len(baseline_tokens) else None
            ct = combined_tokens[i] if i < len(combined_tokens) else None
            if bt != ct:
                print(f"First difference at index {i}: baseline={bt}, combined={ct}")
                break


if __name__ == "__main__":
    main()
