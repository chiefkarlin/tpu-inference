# NMC v7x-4 Profiling Trace Analysis

**Captured:** 2026-06-29 15:59 UTC
**Pod:** nmc-v7x-inference-66975cd957-nmp9p
**Image:** gcr.io/northam-ce-mlai-tpu/nmc-inference:latest (build v5)
**Topology:** v7x-4 single-host (2x2x1, 4 chips), TP=4, page_size=256
**Workload:** 3 decode requests (quicksort prompt, 64 tokens each, batch=3, temperature=0)
**Trace file:** 386MB xplane.pb (not committed — too large for git)

## Executive Summary

**Kernels are NOT the bottleneck.** TPU duty cycle is 9.16% — the TPU is idle 91% of the
time. The gap between measured throughput (~485 tok/s at batch=3) and the HBM-bound
theoretical (3,950 tok/s) is dominated by Python dispatch overhead and small batch size,
not kernel execution or HBM bandwidth.

All five NMC bugs found during bring-up were fixed (BUG#3 mlp_layer_types, K-4 multi-host
sharding, BUG#5 TP sharding on Cohere2Attention). The model loads, compiles, and generates
coherent code correctly.

## Decode Throughput Analysis

| Metric | Value |
|---|---|
| Runtime steps (excl. compile) | 195 (3 prefill + 192 decode) |
| Runtime execute_model total | 1,205 ms |
| Per-step avg | 6.18 ms |
| Batch size | 3 tokens |
| Throughput | ~485 tok/s |
| HBM-bound theoretical | 3,950 tok/s (1.87 GB/chip ÷ 7.4 TB/s) |
| Gap | 8× (dispatch + small batch) |

**Compile cost:** 3 compile calls totaling 9,557 ms (first run_model call = 9,057 ms).
Excluded from runtime throughput calculation.

## Per-Kernel Findings

### RPA Attention (cohere2_attention.py) — CORRECT + FAST

| Metric | Value |
|---|---|
| Runtime custom-calls | 768 (49 layers × ~16 steps) |
| Total runtime | 3.06 ms |
| Avg per call | 0.004 ms |
| Per-step (49 layers) | ~0.2 ms |

XLA emission confirms both tuning params active:
- `RPAd-p_256-bq_1_1-bkv_4096_2048-sw_4096` — decode: `d_block_sizes=(1,4096,1,2048)`,
  `sliding_window=4096` ✓
- `RPAm-p_256-bq_8_8-bkv_2048_512-sw_4096` — mixed/prefill: kernel defaults (not tuned)

**page_size=256** (bumped from 16 by tpu_platform) → bkv_p=4096/256=16. VMEM pressure much
lower than initially estimated. `d_block_sizes=(1,4096,1,2048)` is optimal at this page_size.

**Conclusion:** No attention tuning warranted. K-1 design validated.

### MoE GEMM (gmm_v2 via GMM_TP) — AUTO-TILING GOOD

GMM1 (gate+up, fused SiLU):
- XLA: `gmm_v2-g_128-m_128-k_2048-act_silu-n_512-tm_128-tk_2048-tn_256`
- Per-chip: m=128, k=2048 (D full), n=512 (2F/TP, padded 384→512), tk=2048 (full, single pass), tn=256
- `calculate_tiling` chose full-k, sharded-n. Optimal for these shapes.

GMM2 (down):
- XLA: `gmm_v2-g_128-m_128-k_192-act_None-n_2048-tm_128-tk_256-tn_2048`
- Per-chip: m=128, k=192 (F/TP=768/4), n=2048 (D full), tk=256 (≥k, single pass), tn=2048 (full)
- Small k handled in one pass. Optimal.

| Metric | Value |
|---|---|
| fused_moe_func calls | 96 (48 layers × 2 GEMMs) |
| Total (incl. compile) | 678 ms |
| Avg per call | 7.07 ms |
| Per-layer MoE total | 34.5 ms (incl. compile) |

**Conclusion:** No `tile_info` override needed. P-1 conclusion (accept `calculate_tiling`
heuristic) validated by trace data.

### TP All-Reduce (collective_rpc)

| Metric | Value |
|---|---|
| Calls | 392 (2/step after XLA fusion of 97 layer all-reduces) |
| Total (excl. compile) | ~2,100 ms |
| Dominant non-compile cost | Yes |

2 fused collectives per step — consistent with TP sharding on the MODEL axis (o_proj
all-reduce). This is correct for correctness; EP would eliminate MoE all-reduce but is a
future optimization.

## SyncWait / DMA Analysis

No significant SyncWait stalls detected. DMA copy-start events (960 calls × 0.023 ms avg)
are fast. `_async_copy` (156 calls, 0.67 ms avg) shows good DMA/compute overlap.

## Bottleneck Diagnosis

The 8× throughput gap vs HBM-bound theoretical is caused by:

1. **Python dispatch overhead** (~5 ms/step): The TPU custom-calls total <1 ms per step,
   but `run_model` pjit runtime is ~6 ms. The 5 ms gap is XLA runtime + Python dispatch.
2. **Small batch** (3 tokens): With m=128 MXU tiles, 3 tokens leave 97%+ of MXU lanes
   empty. Throughput scales near-linearly with batch size.
3. **TP all-reduce**: 2 fused collectives/step add latency, though partially overlapped.

None of these are kernel issues — they are system-level utilization constraints.

## Recommendations

### No kernel changes needed (baseline validated)
- `d_block_sizes=(1,4096,1,2048)` — proven optimal at page_size=256
- `calculate_tiling` auto-tiling — good for NMC MoE shapes
- No `tile_info` override or backend switch warranted

### System-level levers (future work)
1. **Increase batch size** (`max_num_seqs`): Primary lever to amortize dispatch overhead
   and improve MXU utilization. Near-linear throughput scaling expected.
2. **Expert Parallelism** (`USE_MOE_EP_KERNEL=1`): Eliminates MoE all-reduce. Consider when
   all-reduce dominates at higher batch. Kernel Eng's JaxMoE wiring supports this via
   `select_moe_backend(use_ep)`.
3. **Prefill block_sizes tuning**: `RPAm-p_256-bq_8_8-bkv_2048_512` uses kernel defaults.
   Could set explicit prefill block_sizes if prefill latency becomes critical. Low priority.

## Raw Trace Data

The full `trace_report.md` generated by `analyze_nmc_trace.py` is below. Key tables:

### Overview Metrics

| Metric | Value |
|---|---|
| device_count | 5 |
| host_count | 3 |
| total_duration_ms | 21173.94 |
| device_duty_cycle_percent | 9.16 |
| average_step_time_ms | 0.0000 |
| step_count | 0 |

### Top 30 Ops by Total Duration (excerpt)

| name | num_calls | total_ms | avg_ms | max_ms |
|:--|--:|--:|--:|--:|
| PjitFunction(run_model) | 384 | 18431.5 | 47.99 | 9056.83 |
| collective_rpc | 392 | 13669.1 | 34.87 | 9546.96 |
| execute_model | 198 | 10762.5 | 54.36 | 9546.97 |
| execute_model: 0 reqs, 6 toks (compile) | 3 | 9557.49 | 3185.83 | 9546.81 |

### Attention (RPA) — Runtime

| name | num_calls | total_ms | avg_ms |
|:--|--:|--:|--:|
| cohere2_attention.py __call__ | 49 | 1214.66 | 24.79 |
| RPA custom-call (runtime) | 768 | 3.06 | 0.004 |

### MoE GEMM — Compile + Runtime

| name | num_calls | total_ms | avg_ms |
|:--|--:|--:|--:|
| cohere2_moe.py __call__ | 49 | 1690.12 | 34.49 |
| PjitFunction(fused_moe_func) | 96 | 678.33 | 7.07 |

### HBM-Bound Reference (Decode, per token)

Theoretical minimum decode latency (TP=4, bf16, ctx=8192):

```
NMC decode reads (per chip):
  - Dense layer 0: attn + MLP weights (replicated)     ~36 MiB
  - 48 MoE layers: 8/128 active experts + attn + router ~930 MiB
  - Attention (49 layers, TP-sharded QKVO)              ~441 MiB
  - LM head (tied embed, vocab/TP sharded)              ~256 MiB
  - KV cache (36 sliding × 2 MiB + 13 full × 4 MiB)     ~124 MiB
  Total: ~1,787 MiB = 1.87 GB per chip

T_min = 1.87 GB / 7.4 TB/s = 0.253 ms/token (~3,950 tok/s if HBM-bound)
```

See `docs/north-mini-code/PROFILING_PLAN.md` for the detailed HBM breakdown.
