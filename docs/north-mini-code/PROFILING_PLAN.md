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

**Verified per-chip decode HBM read** (code-traced through fused_moe_gmm.py:393-394
w1_spec/w2_spec, cohere2_attention.py projections, embed sharding P("model",None),
gmm_v2 IndexMaps group-skipping → only 8/128 active experts read):

| Component                    | Per-chip (TP=4) | Calculation |
|------------------------------|-----------------|-------------|
| MoE GMM1 (8 active experts)  | 12.6 MiB/layer  | 8 × (2048 × 384 × 2B) — col-parallel, 2F/TP=384 |
| MoE GMM2 (8 active experts)  | 6.3 MiB/layer   | 8 × (192 × 2048 × 2B) — row-parallel, F/TP=192, all-reduce |
| MoE router                   | 0.5 MiB/layer   | (2048 × 128 × 2B) — replicated |
| MoE subtotal (48 layers)     | **930 MiB**     | (12.6+6.3+0.5) × 48 |
| Attention Q proj             | 4 MiB/layer     | (2048 × 8×128 × 2B) — N/TP=8 heads |
| Attention K proj             | 0.5 MiB/layer   | (2048 × 1×128 × 2B) — K/TP=1 head |
| Attention V proj             | 0.5 MiB/layer   | same as K |
| Attention O proj             | 4 MiB/layer     | (8×128 × 2048 × 2B) — N/TP=8 heads |
| Attn subtotal (49 layers)    | **441 MiB**     | 9 × 49 |
| Dense MLP (layer 0, replic.) | 36 MiB          | (2048×3072 + 2048×3072 + 3072×2048) × 2B — P() unsharded |
| LM head (tied, vocab/TP)     | 256 MiB         | (262144/4 × 2048 × 2B) — embed sharded P("model",None) |
| KV cache sliding (36 layers) | 72 MiB          | 36 × 4096 × (1×128×2×2B) = 36 × 2 MiB — 1 kv-head/chip |
| KV cache full (13 layers)    | 52 MiB          | 13 × 8192 × 512B = 13 × 4 MiB — at ctx=8192 |
| RMSNorms (49 × 1 norm)       | <1 MiB          | 49 × 2048 × 2B — negligible |
| **Total per-chip (ctx=8192)**| **~1,787 MiB**  | 930+441+36+256+72+52 ≈ 1.75 GiB |

> **Key corrections vs prior estimate (2,719 MB):**
> 1. KV cache: was 392 MB (used 4 kv_heads) → now 124 MB (1 kv-head/chip, TP=4).
> 2. LM head: was 1,074 MB (assumed replicated) → now 256 MB (TP-sharded on vocab).
> 3. Attention: was 334 MB (QKV 2× error) → now 441 MB (9 MiB × 49 layers).
> 4. MoE: was 906 MB → now 930 MB (added 0.5 MiB router/layer, was slightly under).

### 3d. Theoretical minimum decode latency (TP=4, per-chip, ctx=8192)

```
HBM read per chip ≈ 1,787 MiB = 1.874 GB
HBM BW per chip = 7.4 TB/s (Ironwood, confirmed from tpu_specs.json)
T_min ≈ 1.874 GB / 7.4 TB/s ≈ 0.253 ms/token
```

Target decode throughput: ~3,950 tok/s (if HBM-bound at 0.253 ms/tok).
**Actual will be higher** due to: DMA/compute non-overlap, MoE dispatch
overhead, router compute, normalization, sampling, and inter-chip collectives
(48 AllReduces for GMM2 + 49 for O-proj, each ~4 KiB — negligible BW but adds latency).

> At ctx=4096 (shorter context): full-layer KV = 13 × 2 MiB = 26 MiB (vs 52).
> Total ≈ 1,761 MiB → T_min ≈ 0.249 ms (minimal change — weights dominate).
> At ctx=32768: full-layer KV = 13 × 16 MiB = 208 MiB → total ≈ 1,943 MiB → 0.275 ms.
> At ctx=262144 (max): full-layer KV = 13 × 128 MiB = 1,664 MiB → total ≈ 3,399 MiB → 0.481 ms.

### 3e. Prefill theoretical minimums (compute-bound)

Prefill is compute-bound (MXU) for sufficiently long prompts. For prompt length P:
- MoE FLOPs/token: 2 × 8 × (2048×1536 + 768×2048) = 75.5 MFLOP (8 active experts)
- Attention FLOPs/token (approx): 4 × 2048 × 128 × num_kv_heads ≈ 4.2 MFLOP (QK+AV per layer)
- 48 MoE layers + 49 attn layers: ~3.8 GFLOP/token (weights) + ~0.2 GFLOP (attn)
- Total: ~4.0 GFLOP/token
- For P=1024: 4.1 TFLOP, at 2,157 TFLOP/s/chip × 4 chips = 8,628 TFLOP/s → T_min ≈ 0.48 ms
- But prefill is also HBM-bound for weight reads (same weights regardless of P):
  HBM = 1.874 GB/chip (per §3c) → ~0.25 ms floor.
- At P=1024: compute (0.48 ms) > HBM (0.25 ms) → compute-bound.
- Crossover at P ≈ 530 tokens (where 4.0 GFLOP/token × P / 8628 TFLOP/s = 0.25 ms).

> For short prompts (P < ~530), prefill is HBM-bound (same as decode).
> For long prompts (P > ~530), prefill becomes compute-bound.

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
   - Theoretical: 18.9 MiB/layer/chip (8 active experts, TP=4: GMM1 12.6 + GMM2 6.3 MiB).
   - gmm_v2 group-skipping (IndexMaps L256 gm_id_to_group_id): only experts with
     tokens are iterated → 8/128 experts read, not all 128. Verified in code.
   - Check: achieved HBM BW, DMA/compute overlap, expert dispatch overhead.
   - Block sizes: capture calculate_tiling output (tile_m, tile_k, tile_n).
   - Bottleneck hypothesis: small intermediate=768 → small tile_n → MXU
     underutilization in compute-bound (prefill) regime.

2. **RPA v3 attention (sliding layers):** decode with sw=4096.
   - Theoretical KV read: 4096 × 512B = 2 MiB/layer/chip (1 kv-head, TP=4).
   - Kernel CLAMPS to sw internally (kernel.py L388-391 decode, L939-948 prefill).
   - Check: effective KV fetched (should be ≤ sw tokens, confirmed by code).
   - Check: bkv_double_buf VMEM usage (bkv_sz × kv_dim × 2 buffers).
   - With tuned d_block_sizes=(1,4096,1,2048): bkv_sz=4096 (one full sw),
     bkv_csz=2048 (2 compute passes) — 50% VMEM savings vs default bkv_sz=8192.

3. **RPA v3 attention (full layers):** decode with no sw.
   - Theoretical KV read: ctx × 512B/layer/chip. At ctx=8192: 4 MiB/layer.
   - Grows with context: at 256K ctx → 128 MiB/layer → 1,664 MiB total (dominant).
   - Check: bkv_csz VMEM pressure at long contexts.

4. **LM head (tied embedding):** decode vocab projection.
   - Theoretical: 256 MiB/chip (vocab/TP=65536 × 2048 × 2B, TP-sharded P("model",None)).
   - Second-largest decode cost after MoE weights (930 MiB). Verify in trace.
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

  All time columns in picoseconds (÷1e9 → ms, ÷1e6 → µs). Event names resolved
  from `plane.event_metadata[event.metadata_id].name`. DMA transfer *sizes* are
  NOT stored — only durations. `get_hlo_dump()` is a non-functional stub.

  Example queries:
  ```sql
  -- Top-30 ops by total duration (the primary analysis query)
  SELECT name, COUNT(*) AS num_calls, SUM(duration_ps) AS total_ps,
         AVG(duration_ps) AS avg_ps, MAX(duration_ps) AS max_ps
  FROM events GROUP BY name ORDER BY total_ps DESC LIMIT 30;

  -- RPA v3 attention: scope name = RPA{D|P|M}-p_{ps}-bq_{bq}_{bqcsz}-bkv_{bkv}_{bkvcsz}[-sw_{sw}]
  SELECT name, COUNT(*), SUM(duration_ps) AS total_ps, AVG(duration_ps) AS avg_ps
  FROM events WHERE name LIKE '%RPA%' OR name LIKE '%ragged_paged%'
  GROUP BY name ORDER BY total_ps DESC;

  -- MoE GMM kernels
  SELECT name, COUNT(*), SUM(duration_ps) AS total_ps
  FROM events WHERE name LIKE '%gmm%' OR name LIKE '%moe%'
  GROUP BY name ORDER BY total_ps DESC;
  ```

- **Compute vs memory ratio** (`tools/analyze_profile.py::analyze_trace(path)`):
  Uses `xprof.convert.raw_to_tool_data` to get Chrome-trace JSON, finds
  `/device:TPU:0` plane, takes the analysis window as the last two
  `jit_computation` events, sums `dur` of all `SyncWait` events in that window.
  **ratio = SyncWait_total / computation_window**. ratio > 0.5 = memory-bound;
  < 0.3 = compute-bound. This is the key "is this kernel HBM-bound?" metric.
  NOTE: no actual DMA byte counts are extracted — only stall *time*.

- **Per-kernel timing** (`evaluation/xprof_utils.py::extract_xprof_time`):
  Globs `**/*.xplane.pb`, inspects `/device:TPU:0` planes, lines named
  "XLA Modules" or "XLA Ops", averages `duration_ps` over `num_runs`.

- **Overview metrics** (`get_overview_page_metrics`): JSON with
  device_count, total_duration_ms, device_duty_cycle_percent (rough),
  average_step_time_ms, step_count.

- **Ready-to-run analysis script**: `examples/nmc/analyze_nmc_trace.py` —
  combines all the above into a single markdown report. Run inside the Docker
  container: `python3 examples/nmc/analyze_nmc_trace.py <xplane.pb> -o report.md`

- **MaxKernel HITL agent** (`run_hitl_agent.sh`): interactive agent with
  ProfileAgentOrchestrator for DMA/memory transfer analysis, compute vs memory
  ratio, and bottleneck identification with recommendations.

- **JAXBench** (`python -m JAXBench evaluate`): empirical kernel benchmarking
  against baselines. Returns median_ms, tflops, utilization_pct, speedup_vs_baseline.
  Can be used to benchmark individual NMC kernels (MoE GMM, RPA) in isolation.
  NOTE: no JAX/libtpu in the mgmt pod — JAXBench runs on the TPU cluster image.

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

1. Capture the calculate_tiling output for NMC shapes (TP=4, from fused_moe_gmm.py:393-394):
   - GMM1 (gate+up, w1 shape (E,2F,D)=(128,1536,2048), w1_spec=P(None,None,MLP_TENSOR)):
     column-parallel → size_k=D=2048 (full), size_n=2F/TP=1536/4=384 (sharded).
     Per-chip weight: 8 experts × 2048 × 384 × 2B = 12.6 MB/layer.
   - GMM2 (down, w2 shape (E,F,D)=(128,768,2048), w2_spec=P(None,MLP_TENSOR,None)):
     row-parallel → size_k=F/TP=768/4=192 (sharded), size_n=D=2048 (full).
     Per-chip weight: 8 experts × 192 × 2048 × 2B = 6.3 MB/layer.
   - GMM2 reduction: row-parallel → all-reduce (psum) on MLP_TENSOR axis at end.
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
