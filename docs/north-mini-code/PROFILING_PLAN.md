# NMC Profiling Plan (P-4)

Performance Engineer's plan for capturing and analyzing ML Diagnostics traces
of North-Mini-Code inference on v7x-4. This document covers: (1) HBM-bound
theoretical minimums for NMC shapes, (2) per-kernel metrics to extract from
traces, (3) trace capture strategy, and (4) the optimization feedback loop.

---

## 1. Hardware Specs (v7x-4 single host)

| Spec                 | Value                  | Source |
|----------------------|------------------------|--------|
| Chips                | 4                      | topology 2x2x1 |
| Cores/chip           | 2                      | tpu7x arch |
| Total cores          | 8                      | |
| HBM/chip             | 192 GiB                | utils.py:188 + tpu_specs.json |
| HBM/core (JAX dev)   | 96 GiB                 | utils.py:190 |
| Total HBM            | 768 GiB                | |
| HBM BW/chip          | 7.4 TB/s               | MaxKernel tpu_specs.json (Ironwood) |
| MXU peak BF16/chip   | 2,157 TFLOP/s          | MaxKernel tpu_specs.json (Ironwood) |
| MXU peak INT8/chip   | 4,314 TOPS             | MaxKernel tpu_specs.json |
| VMEM/core            | ~64 MB (est.)          | pltpu.get_tpu_info().vmem_capacity_bytes |
| Interconnect         | 3D Torus               | tpu_specs.json |

---

## 2. NMC Model Geometry

- **Architecture:** Cohere2MoeForCausalLM, 30.48B params BF16, ~3B active.
- **Layers:** 49 total. Layer 0 = dense (intermediate=3072). Layers 1-48 = MoE.
- **Attention schedule:** [full, sliding, sliding, sliding] x12 + final full.
  - Full-attn layers: 0, 4, 8, ..., 44, 48 → **13 full** (12 from pattern + final).
  - Sliding-attn layers: 1,2,3, 5,6,7, ..., 45,46,47 → **36 sliding**.
- **MoE:** 128 experts, 8 active/token (topk=8), intermediate=768, hidden=2048,
  gated SiLU, sigmoid routing, norm_topk_prob=false, no shared experts.
- **Attention:** GQA 32 q-heads / 4 kv-heads, head_dim=128, no bias, no QK-norm.
  sliding_window=4096 on sliding layers. RoPE theta=50000 (NeoX/interleaved)
  on sliding + dense layer 0 only (force_rope); NOT on full-attn MoE layers.
- **Vocab:** 262144 x 2048, lm_head tied to embed_tokens, logit_scale=1.0.
- **dtype:** bfloat16. max_position_embeddings=500000 (256K ctx / 64K gen).
- **TP config (assumed):** TP=4 (4 JAX processes, 1 per chip, 2 cores each).
  kv_heads=4 / TP=4 = 1 kv_head per process. q_heads=32 / TP=4 = 8 per process.
  *(Exact TP degree depends on Kernel Eng model; will confirm.)*

---

## 3. HBM-Bound Theoretical Minimums (Decode, per token)

Decode is memory-bound: each generated token requires reading the activated
weight tensors from HBM. The theoretical minimum decode latency is:

```
T_min = total_HBM_bytes_read / HBM_bandwidth
```

### 3a. Per-component weight HBM reads (full model, no TP sharding)

| Component                   | Layers | Shape per expert/token          | Bytes/token (full model) |
|-----------------------------|--------|---------------------------------|--------------------------|
| MoE GMM1 (gate+up)          | 48     | 8 × 2048 × 1536 × 2B           | 48 × 50.3 MB = 2,416 MB  |
| MoE GMM2 (down)             | 48     | 8 × 768 × 2048 × 2B            | 48 × 25.2 MB = 1,208 MB  |
| MoE router                  | 48     | 2048 × 128 × 2B                | 48 × 0.5 MB = 25 MB      |
| Dense MLP gate+up (layer 0) | 1      | 2048 × 6144 × 2B               | 25 MB                    |
| Dense MLP down (layer 0)    | 1      | 3072 × 2048 × 2B               | 12 MB                    |
| Attention QKV proj (per layer) | 49  | 2048 × (32+4+4)×128 × 2B      | 49 × 10.5 MB = 515 MB    |
| Attention O proj (per layer)   | 49  | 4096 × 2048 × 2B              | 49 × 16.8 MB = 823 MB    |
| RMSNorm (per layer, ×2)     | 49     | 2048 × 2B × 2                  | 49 × 8 KB ≈ 0.4 MB       |
| **Subtotal (weights)**      |        |                                 | **~5,024 MB**            |
| LM head (tied embed)        | 1      | 262144 × 2048 × 2B             | 1,074 MB                 |
| **Total weight HBM/token**  |        |                                 | **~6,098 MB ≈ 5.96 GB**  |

### 3b. KV cache reads (attention, per token)

KV per token (bf16, unpacked): 2 × 4 kv_heads × 128 hd × 2B = 4,096 B = 4 KB.
With kv_packing=2: effective 2 KB/token.

| Layer type       | Count | KV read/token (at ctx=4K)        | Total at 4K ctx |
|------------------|-------|----------------------------------|-----------------|
| Sliding (sw=4096)| 36    | min(ctx,4096) × 2KB = 8 MB       | 288 MB          |
| Full             | 13    | ctx × 2KB = 8 MB                 | 104 MB          |
| **Total KV**     |       |                                  | **392 MB**      |

At ctx=32K: full layers read 32K×2KB=64MB each → 13×64=832MB + 36×8=288MB = **1,120 MB**.
At ctx=256K: full layers read 256K×2KB=512MB each → 13×512=6,656MB + 36×8=288MB = **6,944 MB**.

### 3c. Per-chip estimates (TP=4, weights sharded 1/4)

| Component             | Per-chip (TP=4) | Notes |
|-----------------------|-----------------|-------|
| MoE weights           | ~910 MB         | 3,624 MB / 4 (GMM_TP shards hidden dim) |
| Attention weights     | ~334 MB         | 1,338 MB / 4 |
| Dense MLP             | ~9 MB           | 37 MB / 4 |
| LM head               | ~1,074 MB       | Replicated (vocab dim not sharded with TP=4) or sharded |
| KV cache (4K ctx)     | ~392 MB         | kv_heads=1 per process (not sharded further) |
| **Total per-chip**    | **~2,719 MB**   | At 4K context |

> **Caveat:** LM head sharding depends on model implementation. If vocab-parallel
> (sharded across TP), per-chip = 1,074/4 = 268 MB. If replicated, 1,074 MB.
> This is the single largest decode cost component — verify in trace.

### 3d. Theoretical minimum decode latency (TP=4, per-chip, 4K ctx)

```
HBM read per chip ≈ 2,719 MB = 2.719 GB
HBM BW per chip = 7.4 TB/s (Ironwood, confirmed from tpu_specs.json)
T_min ≈ 2.719 GB / 7.4 TB/s ≈ 0.37 ms/token
```

Target decode throughput: ~2,700 tok/s (if HBM-bound at 0.37 ms/tok).
**Actual will be higher** due to: DMA/compute non-overlap, MoE dispatch
overhead, router compute, normalization, sampling, and inter-chip collectives.

### 3e. Prefill theoretical minimums (compute-bound)

Prefill is compute-bound (MXU). For prompt length P tokens:
- MoE FLOPs/token: 2 × 8 × (2048×1536 + 768×2048) = 75.5 MFLOP
- Attention FLOPs/token (approx): 4 × 2048 × 128 × num_kv_heads ≈ 4.2 MFLOP (QK+AV per layer)
- 48 MoE layers + 49 attn layers: ~3.8 GFLOP/token (weights) + ~0.2 GFLOP (attn)
- Total: ~4.0 GFLOP/token
- For P=1024: 4.1 TFLOP, at 2,157 TFLOP/s/chip × 4 chips = 8,628 TFLOP/s → T_min ≈ 0.48 ms
- But prefill is also HBM-bound for weight reads (same weights regardless of P):
  HBM = 2.719 GB/chip, same as decode → ~0.37 ms floor.

> For short prompts (P < ~256), prefill is HBM-bound (same as decode).
> For long prompts (P > ~1024), prefill becomes compute-bound.

---

## 4. Per-Kernel Metrics to Extract from Traces

### 4a. Trace capture method

Use `jax.profiler.trace()` around the inference loop, or set
`VLLM_XLA_CACHE_PATH` and capture via the ML Diagnostics SDK/CLI:

```python
import jax.profiler
with jax.profiler.trace("/tmp/jax_trace", create_perfetto_trace=True):
    # warmup + measured decode steps
    for _ in range(warmup):
        model.generate(...)
    for _ in range(measured):
        model.generate(...)
```

Alternative: set env `JAX_TRACE_START_GPU=10 JAX_TRACE_STOP_GPU=20` to capture
steps 10-20 automatically.

### 4b. Key metrics per kernel (from Perfetto trace)

| Metric                    | How to measure                          | Target |
|---------------------------|-----------------------------------------|--------|
| Kernel wall time          | Trace event duration                    | Compare to T_min |
| HBM bytes read            | DMA event sizes (HBM→VMEM transfers)    | Compare to theoretical |
| Achieved HBM BW           | HBM bytes / kernel time                 | >70% of peak (~5.2 TB/s) |
| MXU utilization           | MXU compute events / kernel time        | Decode: <5% (memory-bound); Prefill: >50% |
| DMA/compute overlap       | Overlap of DMA and MXU events in trace  | >60% overlap |
| VMEM utilization          | Max VMEM allocated (from Pallas compile)| <90% of capacity |
| Pipeline stalls           | Gaps between compute events             | Minimize |
| Inter-chip collectives    | ICI transfer events (all-reduce, etc.)  | Compare to compute time |

### 4c. Kernels of interest (priority order)

1. **MoE GMM (gmm_v2):** gate/up + down GEMMs. Decode is HBM-bound (weight read).
   - Theoretical: 72 MB/layer/token (8 experts, full model). 
   - Check: achieved HBM BW, DMA/compute overlap, expert dispatch overhead.
   - Block sizes: capture calculate_tiling output (tile_m, tile_k, tile_n).
   - Bottleneck hypothesis: small intermediate=768 → small tile_n → MXU
     underutilization in compute-bound (prefill) regime.

2. **RPA v3 attention (sliding layers):** decode with sw=4096.
   - Theoretical KV read: 4096 × 2KB = 8 MB/layer/token.
   - Check: effective KV fetched (should be ≤ sw tokens, not bkv_sz).
   - Check: bkv_double_buf VMEM usage (bkv_sz × kv_dim × 2 buffers).
   - With default bkv_sz=8192: VMEM for 8192 tokens allocated but only 4096 used.
   - With tuned bkv_sz=4096: 50% VMEM savings, same HBM read.

3. **RPA v3 attention (full layers):** decode with no sw.
   - Theoretical KV read: ctx × 2KB. Grows with context length.
   - Check: bkv_csz VMEM pressure at long contexts (256K → 512 MB KV read).
   - This is the dominant cost at long context — verify HBM BW achieved.

4. **LM head (tied embedding):** decode vocab projection.
   - Theoretical: 1,074 MB weight read (full) or ~268 MB (TP-sharded).
   - This may be the single largest decode cost — verify in trace.
   - Check: is it GEMM (efficient) or gather (inefficient)?

5. **Router / top-k selection:** MoE gating.
   - Small (2048×128 weight = 0.5 MB) but has scatter/gather overhead.
   - Check: dispatch latency, any serialization.

6. **RMSNorm:** 49 layers × 2 = 98 norms per step.
   - Tiny (8 KB each) but many kernel launches → launch overhead.
   - Check: are they fused with adjacent ops?

---

## 5. Trace Capture Strategy

### Phase 1: Baseline (post P-3 e2e run)

1. **Short-prompt decode trace:** 1 request, 32-token prompt, 64-token decode.
   Capture steps 10-20 (skip warmup). Focus: per-kernel wall time, HBM BW.
2. **Long-prompt decode trace:** 1 request, 4K-token prompt, 64-token decode.
   Focus: full-attn layer KV scaling, sliding vs full attention comparison.
3. **Prefill trace:** 1 request, 1024-token prompt. Focus: MXU utilization,
   MoE GMM tile sizes, compute-bound behavior.

### Phase 2: Tuned (post P-2 attention block_sizes tuning)

4. **Repeat short-prompt decode** with d_block_sizes=(1,4096,1,2048).
   Compare VMEM usage, DMA/compute overlap vs baseline.
5. **Repeat long-prompt decode** — verify VMEM headroom improvement.

### Phase 3: MoE deep-dive (if MoE proven bottleneck)

6. **MoE-focused trace:** isolate gmm_v2 kernel. Capture tile sizes from
   calculate_tiling. Measure achieved BW vs theoretical for 8-expert dispatch.
7. If justified, prototype tile_info override and re-profile.

---

## 6b. Trace Analysis Tooling (MaxKernel)

The `accelerator-agents/MaxKernel` toolkit provides offline xplane.pb analysis:

- **`offline_tools.py::load_xplane_and_query(xplane_path, sql_query)`**: loads an
  xplane.pb trace into an in-memory SQLite DB and runs SQL queries. Schema:
  - `planes (id, name)` — TPU planes (hosts/devices)
  - `lines (id, plane_id, display_id, name, timestamp_ns)` — trace lines/streams
  - `events (plane_id, line_id, name, offset_ps, duration_ps, start_ps, end_ps)` — kernel/DMA events

  Example queries:
  ```sql
  -- Top-10 longest kernel events
  SELECT name, COUNT(*) as calls, AVG(duration_ps)/1e6 as avg_ms
  FROM events GROUP BY name ORDER BY AVG(duration_ps) DESC LIMIT 10;

  -- Total time in RPA attention kernels
  SELECT SUM(duration_ps)/1e6 as total_ms FROM events
  WHERE name LIKE '%ragged_paged_attention%';

  -- MoE GMM kernel timings
  SELECT name, COUNT(*), SUM(duration_ps)/1e6 as total_ms
  FROM events WHERE name LIKE '%gmm%' GROUP BY name;
  ```

- **MaxKernel HITL agent** (`run_hitl_agent.sh`): interactive agent with
  ProfileAgentOrchestrator for DMA/memory transfer analysis, compute vs memory
  ratio, and bottleneck identification with recommendations.

- **JAXBench** (`python -m JAXBench evaluate`): empirical kernel benchmarking
  against baselines. Returns median_ms, tflops, utilization_pct, speedup_vs_baseline.
  Can be used to benchmark individual NMC kernels (MoE GMM, RPA) in isolation.

---

## 6. Optimization Feedback Loop

```
Profile baseline → identify top-3 time-consuming kernels
  → compute theoretical minimum for each
  → compute efficiency gap (actual / theoretical)
  → if gap > 1.5x: investigate DMA/compute overlap, VMEM pressure, tiling
  → propose tuning (block_sizes for attention, tile_info for MoE)
  → Kernel Eng implements, I re-profile
  → iterate until gap < 1.3x or diminishing returns
```

### Efficiency gap thresholds

| Gap (actual/theoretical) | Assessment                | Action |
|--------------------------|---------------------------|--------|
| < 1.3x                   | Well-optimized            | Accept, move to next kernel |
| 1.3x - 2.0x              | Sub-optimal, tunable      | Investigate tiling/overlap |
| > 2.0x                   | Significant bottleneck    | Deep analysis, consider kernel changes |

---

## 7. P-1 Follow-up: MoE tile_info Tuning Plan (post-baseline)

**Current state:** GMM_TP backend → gmm_v2 → calculate_tiling (pure VMEM-fit
heuristic). No tuned table. gmm_wrapper hardcodes tile_info=default.

**If MoE is a proven bottleneck after baseline profiling:**

1. Capture the calculate_tiling output for NMC shapes:
   - GMM1: size_m=tokens, size_k=2048/TP, size_n=1536/TP (or full, depends on shard)
   - GMM2: size_m=tokens, size_k=768, size_n=2048/TP
   - Record tile_m, tile_k, tile_n from the trace compile logs.

2. Identify whether tile_n is too small (MXU underutilization) or tile_k is
   too small (excessive accumulation overhead). For NMC intermediate=768:
   - tile_k starts at align_to(768, num_lanes). If num_lanes=128: tile_k=768 (full K in one pass).
   - tile_n starts at full, shrinks to fit VMEM. For bf16 unquantized: should be large.

3. Prototype: write a custom TileFn that returns TileSizes with larger tile_n
   (if VMEM allows) or smaller tile_k (if accumulation is the bottleneck).
   Wire through gmm_wrapper → gmm_v2 via tile_info param.

4. Re-profile and compare. If improvement > 15%, propose as permanent change
   (requires orchestrator approval since it touches shared gmm_wrapper infra).

**Alternative:** switch to FUSED_MOE backend (USE_MOE_EP_KERNEL=1 + EP) which
engages the fused_moe/v1 tuned table. Only viable if EP is enabled (needs
multi-chip EP topology, not available on single-host v7x-4 TP-sharded).
