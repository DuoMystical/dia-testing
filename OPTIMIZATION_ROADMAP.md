# Dia2 Streaming Server Optimization Roadmap

## Current Performance

| Metric | Value |
|--------|-------|
| Cold start | ~17s |
| Frame rate | ~12.5 Hz (80ms of audio per frame) |
| Step time | ~36ms |
| RTF (real-time factor) | 0.45 (2.2x faster than real-time) |
| 10s audio generation | ~4.5s |

### Cold Start Breakdown (Current)

| Component | Time |
|-----------|------|
| Python imports | ~9s |
| Model download + GPU load (parallel) | ~6s |
| Tokenizer | ~1s |
| Mimi codec | ~1.5s |
| **Total** | **~17s** |

### Inference Breakdown (Current)

| Component | Time |
|-----------|------|
| Transformer step | ~10ms |
| Depformer step | ~26ms |
| Python overhead | ~3ms |
| **Total per step** | **~36ms** |

---

## Optimization Option 1: Full Rewrite (Rust + ONNX + TensorRT)

### Expected Performance

| Metric | Current | Optimized |
|--------|---------|-----------|
| Cold start | 17s | ~7s |
| Step time | 36ms | ~25ms |
| RTF | 0.45 | 0.31 |
| 10s audio generation | 4.5s | ~3.1s |

### Cold Start Breakdown (Optimized)

| Component | Time |
|-----------|------|
| Rust runtime init | ~0.1s |
| Model download + GPU load | ~6s |
| Tokenizer (rust-tokenizers) | ~0.2s |
| Mimi codec (ONNX) | ~0.5s |
| **Total** | **~7s** |

### Components to Convert

| Component | Difficulty | Notes |
|-----------|------------|-------|
| ONNX export (Transformer) | **Hard** | Autoregressive KV cache is tricky to export |
| ONNX export (Depformer) | **Hard** | Custom architecture, multi-step generation |
| TensorRT optimization | Medium | Tooling exists, but tuning needed |
| Mimi codec to ONNX | Medium | Need to preserve streaming behavior |
| Rust ONNX Runtime integration | Medium | Replace Python bridge with `ort` crate |
| Port generation logic to Rust | Medium | State machine, sampling, delay patterns |
| Tokenizer | Easy | `rust-tokenizers` or `tokenizers` crate |

### The Hard Part: Autoregressive ONNX Export

The model uses KV cache that updates each step. Options:

1. **External cache management** - Export model to accept/return cache tensors, manage in Rust
2. **ONNX Scan operator** - Complex but keeps everything in ONNX graph
3. **Single-step export** - Export as one-step model, loop in Rust

### Time Estimate

| Phase | Hours |
|-------|-------|
| ONNX export + validation | 8-15 |
| TensorRT conversion | 3-6 |
| Mimi codec export | 4-8 |
| Rust integration | 6-12 |
| Testing/debugging | 6-12 |
| **Total** | **27-53 hours** |

Roughly **1-2 weeks full-time** or **3-5 weeks part-time**.

---

## Optimization Option 2: Simpler Path (Keep PyTorch)

### Expected Performance

| Metric | Current | Optimized |
|--------|---------|-----------|
| Cold start | 17s | ~12s |
| Step time | 36ms | ~30ms |
| RTF | 0.45 | 0.38 |

### Changes Required

1. **torch.compile()** - Add `@torch.compile` to model forward passes (~20% inference speedup)
2. **Slim Docker image** - Use smaller base image, remove unnecessary packages
3. **Pre-download model into image** - Bake weights into Docker image (larger image, but no download on start)
4. **Optimize Python imports** - Lazy imports, remove unused dependencies

### Time Estimate

**4-8 hours total**

---

## Concurrency Considerations

For multiple concurrent requests on the same server:

| Approach | Complexity | Latency Impact | Throughput |
|----------|------------|----------------|------------|
| 1 GPU, 1 request at a time | Low | None | 1x |
| 1 GPU, interleaved requests | Medium | Increased | ~1.2-1.5x |
| 2 GPUs, 1 request each | Low | None | 2x |
| 1 GPU, batched inference | High | Variable | ~1.5-2x |

**Recommendation**: Horizontal scaling (multiple GPUs/instances) beats trying to multiplex concurrent requests on one GPU. The autoregressive nature means one GPU can only generate ~27 steps/second regardless of concurrency.

---

## Recommendation

For autoscaling use case:

1. **Short term (Option 2)**: Optimize Docker + torch.compile for ~12s cold start
2. **Long term (Option 1)**: Full Rust+ONNX+TensorRT rewrite if cold start is critical

The full rewrite saves ~10s on cold start and ~1.4s per 10s of audio. Worth it if you're doing high-volume autoscaling where those seconds add up.
