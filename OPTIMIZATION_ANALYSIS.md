# Dia2 Streaming TTS Optimization Analysis

## Current Performance Baseline (from logs)

```
Total: 5175ms, Steps: 116, Audio: 7920ms (1.5x realtime)

CUDA graph capture: 25ms
Transformer steps:  5ms (0.0ms/step)
Text sampling:      3762ms (32.4ms/step)  ← SUSPICIOUS
Machine process:    1ms (0.0ms/step)
Audio sampling:     58ms (0.5ms/step)
Depformer stages:   1124ms (9.7ms/step)
Audio decode:       173ms
Other/overhead:     27ms
```

## The Core Problem: Implicit Synchronization

The timing shows "text sampling" at 32.4ms/step, but this is **misleading**. Here's why:

```
Step N timeline:
1. Transformer runs (launches async GPU kernels)
2. Text sampling calls .item()
   → .item() BLOCKS until ALL pending GPU work completes
   → This includes the transformer that JUST launched
3. Timer attributes all that wait time to "text sampling"
```

So the 32.4ms isn't sampling - it's **waiting for the transformer + depformer from current step**. The actual sampling (softmax + multinomial) should be ~0.1ms.

## Sync Points Analysis

Per step, these are the CPU-GPU sync points:

| Operation | Sync Type | Can Avoid? |
|-----------|-----------|------------|
| `text_token = sample().item()` | Explicit | Maybe |
| `main_tokens.fill_(main_token)` | None (CPU→GPU) | No |
| `audio_buf[:, 0, t+1] = codebook_token` | None (GPU→GPU) | Already optimal |
| `audio_buf[:, stage+1, t+1] = stage_token` | None (GPU→GPU) | Already optimal |

The **only explicit sync** is `.item()` for the text token.

## Why We Need That Sync

The state machine needs the text token value to decide:
1. Should we consume the next text entry?
2. Should we force padding?
3. What's the output token?

Dependency chain:
```
transformer → text_logits → sample → .item() → machine.process() → step_tokens → depformer
                                        ↑
                                   CAN'T SKIP THIS
```

## Potential Solutions

### Option A: GPU-side State Machine (Complex but Maximum Speedup)

Rewrite the state machine logic as tensor operations:

```python
# Instead of Python state machine:
# text_token_tensor stays on GPU
# Use torch.where() for conditionals
# Track state as GPU tensors

is_new_word = (text_token_tensor == token_ids.new_word)
should_force_pad = (forced_padding_tensor > 0) | (pending_count_tensor > 0)
effective_token = torch.where(should_force_pad, pad_tensor, text_token_tensor)
```

**Pros:** Eliminates the sync entirely
**Cons:** Complex rewrite, state machine has Python deques that are hard to tensorize

### Option B: Speculative Execution (Medium Complexity)

Run the depformer speculatively assuming `pad`, then correct if wrong:

```python
# Assume pad (most common case)
step_tokens[:, 0, 0] = token_ids.pad

# Launch depformer (async)
run_depformer_async()

# Meanwhile, sync and check actual token
text_token = sample().item()
main_token, aux_token, _ = machine.process(...)

# If we guessed wrong, recompute (rare)
if main_token != token_ids.pad:
    rerun_depformer()
```

**Pros:** Fast path for common case (pad is most frequent)
**Cons:** Complexity, wasted work when wrong

### Option C: Pipelined Execution with CUDA Streams (Medium Complexity)

Overlap step N's audio decode with step N+1's transformer:

```python
compute_stream = torch.cuda.Stream()
decode_stream = torch.cuda.Stream()

for step in range(max_steps):
    with torch.cuda.stream(compute_stream):
        run_transformer()
        run_depformer()

    if should_emit_chunk:
        with torch.cuda.stream(decode_stream):
            decode_audio(chunk)  # Parallel with next transformer
```

**Pros:** Overlaps audio decode (~173ms total) with generation
**Cons:** Doesn't address the main sync issue

### Option D: Batch Multiple Steps (Significant Rewrite)

Process multiple text tokens before syncing:

```python
for i in range(BATCH_SIZE):
    text_logits_batch[i] = run_transformer_step()

torch.cuda.synchronize()  # Single sync for N steps
text_tokens = sample_batch(text_logits_batch).tolist()

for i, token in enumerate(text_tokens):
    machine.process(...)
```

**Pros:** Amortizes sync cost over N steps
**Cons:** State machine is sequential (step N affects step N+1)

## Implementation Plan

### Phase 1: Accurate Timing (IMPLEMENTED)
Add explicit `torch.cuda.synchronize()` after each major section to get real breakdown.

### Phase 2: Async Audio Decode (IMPLEMENTED)
Use separate CUDA stream for audio decode to overlap with generation.

### Phase 3: Based on Real Numbers
- If transformer is bottleneck → limited options (already using CUDA graphs)
- If depformer is bottleneck → look at batching/fusion
- If sync overhead is real → implement GPU state machine (Option A)

## Python vs Native Code

For this workload, Python is NOT the main bottleneck:
- Neural network math runs on GPU via CUDA kernels
- PyTorch overhead is minimal for GPU-bound operations
- The sync issue would exist in any language

Native code would help with:
- Loop iteration overhead (~0.1ms/step)
- Memory allocation patterns
- Total savings: ~100-200ms on 5s generation (~3-4%)

## Notes on Mimi Codec Constraints

- Frame rate: 12.5Hz (80ms per frame)
- max_delay: 18 frames (from model's delay_pattern)
- Minimum chunk size: 19 frames = 1520ms

This is a fundamental limitation of the codec architecture, not something we can optimize away.
