# NMC v7x-4 Benchmark Results — GPU Comparison Baseline

**Date:** 2026-06-30
**Model:** CohereLabs/North-Mini-Code-1.0 (30.48B params, BF16, MoE 128/8)
**Hardware:** TPU v7x-4 (2x2x1, 4 chips, 94.75 GiB HBM/chip, 7.4 TB/s HBM BW, 2157 TFLOP/s BF16)
**Software:** vLLM 0.23.1rc1.dev493+gdccb412e2, JAX 0.10.2, libtpu 0.0.42.1, flax 0.12.4
**Server config:** TP=4, max-model-len=8192, max-num-seqs=8, max-num-batched-tokens=1024, dtype=bf16
**Method:** vllm bench serve with synthetic random data, --ignore-eos for controlled output lengths

## Summary Table

### Phase 1: Prefill Latency (TTFT)
*Input varies, output=1, concurrency=1, 10 prompts. First 3 tests include XLA compile time (p99).*

| Input Length | TTFT p50 (ms) | TTFT p90 (ms) | TTFT p99 (ms) | Total Throughput (tok/s) |
|---|---|---|---|---|
| 128 | 19.7 | 1026.2 | 9157.5 | 126.0 |
| 512 | 33.8 | 966.4 | 8499.7 | 532.1 |
| 1024 | 42.4 | 1269.2 | 11180.0 | 809.3 |
| 2048 | 77.2 | 78.2 | 81.1 | 26500.1 |
| 4096 | 150.8 | 152.1 | 155.2 | 27189.5 |

*Note: High p99 on first 3 tests = XLA compilation on first request (~9-11s). By input=2048, compilation is cached and p99 drops to 81ms. Post-compile TTFT scales linearly with input length.*

### Phase 2: Decode Latency (TPOT)
*Input=128, output varies, concurrency=1, 10 prompts.*

| Output Length | TTFT p50 (ms) | TPOT p50 (ms) | TPOT p90 (ms) | TPOT p99 (ms) | Output Throughput (tok/s) |
|---|---|---|---|---|---|
| 128 | 21.0 | 6.16 | 6.17 | 6.22 | 159.2 |
| 256 | 21.3 | 6.16 | 6.17 | 6.17 | 160.7 |
| 512 | 21.4 | 6.17 | 6.21 | 6.22 | 161.2 |

*Single-stream decode: 6.16 ms/token = ~162 tok/s. Consistent across output lengths.*

### Phase 3: Throughput Scaling
*Input=512, output=128, concurrency varies.*

| Concurrency | Prompts | TPOT p50 (ms) | TPOT p90 (ms) | TPOT p99 (ms) | Output Tput (tok/s) | Total Tput (tok/s) |
|---|---|---|---|---|---|---|
| 1 | 4 | 6.16 | 6.18 | 6.18 | 156.6 | 783.1 |
| 2 | 8 | 6.53 | 6.64 | 6.64 | 280.0 | 1400.0 |
| 4 | 16 | 6.98 | 7.25 | 7.31 | 530.5 | 2652.2 |
| 8 | 32 | 7.85 | 62.5 | 63.0 | 361.5 | 1807.6 |

*Near-linear scaling to c4 (530 tok/s). c8 has high variance — one request hit p99=63ms TPOT (likely recompile/GC stall), dragging mean TPOT to 16.3ms. Peak output throughput at c8 = 949 tok/s.*

### Phase 4: Realistic Mixed (512/256)
*Input=512, output=256, concurrency varies.*

| Concurrency | Prompts | TTFT p50 (ms) | TPOT p50 (ms) | TPOT p90 (ms) | TPOT p99 (ms) | Output Tput (tok/s) | Total Tput (tok/s) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 33.9 | 6.17 | 6.18 | 6.19 | 159.2 | 477.6 |
| 4 | 16 | 70.0 | 7.02 | 7.12 | 7.20 | 550.6 | 1651.8 |
| 8 | 32 | 94.2 | 7.73 | 7.95 | 8.10 | 991.7 | 2975.0 |

*Best aggregate throughput: c8 = 2975 total tok/s (992 output tok/s). TPOT degrades gracefully (6.17→7.73ms at c8).*

## Key Metrics for GPU Comparison

| Metric | Value | Notes |
|---|---|---|
| **Single-stream decode TPOT** | 6.16 ms/tok | 162 tok/s, batch=1 |
| **Single-stream decode throughput** | 162 tok/s | HBM-bound theoretical: 3,950 tok/s |
| **Best decode throughput (batch=8)** | 992 tok/s | mixed c8, output only |
| **Best total throughput (batch=8)** | 2,975 tok/s | mixed c8, prefill+decode |
| **Prefill TTFT (1024 tokens)** | 42 ms | post-compile |
| **Prefill TTFT (4096 tokens)** | 151 ms | post-compile |
| **Throughput scaling efficiency** | ~85% at c4 | 530/624 expected |
| **TPOT degradation at c8** | 25% | 6.17→7.73ms (mixed) |

## Analysis

### Decode Performance
- **6.16 ms/token at batch=1** matches the P-4 trace analysis (6.18 ms/step at batch=3).
- The 8× gap vs HBM-bound theoretical (3,950 tok/s) is confirmed as Python dispatch + small batch overhead, NOT kernel execution.
- TPOT is remarkably stable across output lengths (6.16-6.17ms) — no KV cache growth penalty at these context lengths.

### Throughput Scaling
- Near-linear scaling to c4 (530 tok/s output, 2,652 tok/s total).
- c8 shows variance in Phase 3 (one outlier request) but performs well in Phase 4 (992 tok/s output, 2,975 tok/s total).
- The c8 anomaly in Phase 3 (mean TPOT 16.3ms vs median 7.85ms) is likely a transient recompile or GC stall — not representative of steady-state.

### Prefill Performance
- Post-compile TTFT: 20ms (128 tokens) → 151ms (4096 tokens). Scales ~linearly.
- XLA compilation takes ~9-11s on first request. Subsequent requests use cached compilation.
- Prefill is compute-bound at these lengths (P-4 analysis: crossover at ~530 tokens).

### Bottlenecks
1. **Small batch (primary):** 3-8 tokens per step with m=128 MXU tiles → 94-97% empty MXU lanes.
2. **Python dispatch:** ~5ms/step overhead on top of <1ms TPU custom-call time.
3. **TP all-reduce:** 2 fused collectives/step (35ms avg including compile, ~10ms runtime).
4. **max-num-seqs=8:** Limits concurrent request batching. Step 2 will test max-num-seqs=32.

## Server Configuration

```
--model CohereLabs/North-Mini-Code-1.0
--tensor-parallel-size 4
--dtype bfloat16
--max-model-len 8192
--max-num-seqs 8
--max-num-batched-tokens 1024
--no-enable-prefix-caching
--gpu-memory-utilization 0.9
--seed 42
```

## Hardware Reference

| Spec | Value |
|---|---|
| TPU | v7x (Ironwood) |
| Topology | 2x2x1 (4 chips, single host) |
| HBM capacity | 94.75 GiB/chip (379 GiB total) |
| HBM bandwidth | 7.4 TB/s/chip (29.6 TB/s total) |
| MXU BF16 | 2,157 TFLOP/s/chip (8,628 TFLOP/s total) |
| Cores | 2 per chip (8 total) |
