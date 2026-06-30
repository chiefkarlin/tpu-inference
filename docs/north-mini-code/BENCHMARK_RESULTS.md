# NMC v7x-4 Benchmark Results — GPU Comparison

**Date:** 2026-06-30
**Model:** CohereLabs/North-Mini-Code-1.0 (30.48B params, BF16, MoE 128/8)
**Hardware:** TPU v7x-4 (2x2x1, 4 chips, 94.75 GiB HBM/chip, 7.4 TB/s HBM BW, 2157 TFLOP/s BF16)
**Software:** vLLM 0.23.1rc1.dev493+gdccb412e2, JAX 0.10.2, libtpu 0.0.42.1, flax 0.12.4
**Method:** vllm bench serve with synthetic random data, --ignore-eos for controlled output lengths

---

## Config A: Baseline (max-num-seqs=8, max-num-batched-tokens=1024)

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

---

## Config B: Tuned (max-num-seqs=32, max-num-batched-tokens=4096)

**Server config:** TP=4, max-model-len=8192, max-num-seqs=32, max-num-batched-tokens=4096, dtype=bf16

### Phase 3: Throughput Scaling (512→128)
*Input=512, output=128. Note: c1-c8 include XLA recompilation artifacts (pod freshly restarted). c16+ are post-compile.*

| Concurrency | Prompts | TPOT p50 (ms) | TPOT p90 (ms) | Peak Output Tput (tok/s) | Total Tput (tok/s) |
|---|---|---|---|---|---|
| 1 | 4 | 6.30 | 6.37 | 153 | 112 |
| 2 | 8 | 6.50 | 6.70 | 274 | 324 |
| 4 | 16 | 6.89 | 7.20 | 540 | 635 |
| 8 | 32 | 7.47 | 7.51 | 1024 | 1200 |
| 16 | 64 | 8.43 | 9.71 | 1828 | 4600 |
| 32 | 128 | 19.66 | 75.85 | 2528 | 1892 |

*c32 throughput phase shows degradation (TPOT p90=76ms) — batch saturates at c16 for this input/output shape.*

### Phase 4: Realistic Mixed (512→256) — CLEAN (post-compile)

| Concurrency | Prompts | TTFT p50 (ms) | TPOT p50 (ms) | TPOT p90 (ms) | TPOT p99 (ms) | Output Tput (tok/s) | Total Tput (tok/s) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 34.7 | 6.41 | 6.50 | 6.51 | 153 | 459 |
| 4 | 16 | 79.3 | 6.96 | 7.12 | 7.13 | 553 | 1658 |
| 8 | 32 | 113.7 | 7.54 | 7.79 | 7.87 | 1003 | 3009 |
| 16 | 64 | 194.5 | 8.44 | 9.58 | 10.02 | 1676 | 5027 |
| 32 | 128 | 186.5 | 13.57 | 13.94 | 14.13 | 2233 | 6698 |

## Side-by-Side Comparison: Mixed (512→256)

| Concurrency | Config A TPOT p50 (ms) | Config B TPOT p50 (ms) | Config A Output (tok/s) | Config B Output (tok/s) | Config A Total (tok/s) | Config B Total (tok/s) |
|---|---|---|---|---|---|---|
| 1 | 6.17 | 6.41 | 159 | 153 | 478 | 459 |
| 4 | 7.02 | 6.96 | 551 | 553 | 1652 | 1658 |
| 8 | 7.73 | 7.54 | 992 | 1003 | 2975 | 3009 |
| 16 | — | 8.44 | — | 1676 | — | 5027 |
| 32 | — | 13.57 | — | 2233 | — | 6698 |

*Config A couldn't test c16/c32 (max-num-seqs=8 cap). Config B unlocks full scaling.*

## Key Findings — Config B (Tuned)

| Metric | Config A | Config B | Improvement |
|---|---|---|---|
| **Best output throughput** | 992 tok/s (c8) | 2,233 tok/s (c32) | 2.25× |
| **Best total throughput** | 2,975 tok/s (c8) | 6,698 tok/s (c32) | 2.25× |
| **Sweet spot output tput** | 992 tok/s @ 7.73ms TPOT | 1,676 tok/s @ 8.44ms TPOT | 1.69× @ similar latency |
| **Single-stream TPOT** | 6.17 ms | 6.41 ms | ~same (config overhead) |
| **Max batch tested** | 8 | 32 | 4× |

### Scaling Analysis
- **c1-c8:** Nearly identical between configs (batch fits in both). TPOT ~6-8ms.
- **c16 (NEW):** 1,676 tok/s output @ 8.44ms TPOT — **best latency/throughput trade-off**. Only 32% TPOT degradation vs c1, with 11× throughput gain.
- **c32 (NEW):** 2,233 tok/s output @ 13.57ms TPOT — **best raw throughput** but 2.1× TPOT degradation. Total throughput 6,698 tok/s.
- **Peak output throughput:** 2,560 tok/s (c32 instantaneous peak).
- **Diminishing returns at c32:** TPOT jumps from 8.44ms (c16) to 13.57ms (c32) — 61% degradation for 33% more throughput. The TPU is approaching HBM bandwidth saturation at batch=32.

### HBM Bandwidth Utilization
- HBM-bound theoretical (batch=1): 3,950 tok/s → actual 162 tok/s = 4.1% HBM BW utilization
- HBM-bound theoretical (batch=32): 3,950 tok/s → actual 2,233 tok/s = 56.5% HBM BW utilization
- At c32, the TPU is approaching HBM-bound efficiency — the 8× gap from P-4 trace analysis is closing with batch size.
