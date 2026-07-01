# NMC TPU v7x Performance Optimization Report

**Date:** 2026-06-30
**Model:** CohereLabs/North-Mini-Code-1.0 (30.48B params, BF16, MoE 128/8)
**Hardware:** TPU v7x-4 (2x2x1, 4 chips) vs H200 (4× GPU)
**Baseline:** Config B (TP=4, max-num-seqs=32, max-num-batched-tokens=4096)

---

## Executive Summary

Five optimization phases were attempted to close the 1.5–2× performance gap between TPU v7x and H200 GPU. **Two optimizations succeeded and stack: async_scheduling (+8-29%) and single_step_decode (fused dispatch).** The combined configuration delivers **2× throughput over Config B at c32** and — critically — **the TPU now BEATS H200 at c16 and c32**. The gap narrowed from 1.5–2.5× to **parity/surpassing at c16+**. The TPU's 1.54× HBM bandwidth and 2.18× BF16 compute advantage is finally realized at high batch through fused dispatch + async D2H overlap.

---

## Optimization Attempts

### Phase 1: continue_decode (Decode Dispatch Fusion) — ❌ FAILED

**Hypothesis:** Fusing N decode steps into one jitted `while_loop` would eliminate per-step pjit dispatch overhead (the 5ms/step from P-4 trace).

**Result:** Continue_decode is **slower** across all batch sizes:

| Concurrency | Config B TPOT | continue_decode TPOT | Delta |
|-------------|--------------|----------------------|-------|
| 1 | 6.41ms | 7.20ms | +12% |
| 8 | 7.54ms | 8.53ms | +13% |
| 16 | 8.44ms | 8.70ms | +3% |
| 32 | 13.57ms | 14.81ms | +9% |

**Root cause:** The while_loop overhead + sampling_metadata padding (computing logits for 16 positions instead of 8) exceeds the dispatch savings. The 5ms/step overhead is XLA runtime scheduling, not pure Python dispatch — continue_decode can't eliminate it.

**Bugs found & fixed during development:**
- Bug 1 (crash): Token padding vs request padding mismatch in `_execute_continue_decode`. Fix: slice `init_tokens` to align with precompile shapes.
- Bug 2 (correctness): Garbled output from init_tokens slicing dropping token padding. Fix: pad sampling_metadata to token-padded batch size.

### Phase 2: Prefill Block Size Sweep — ❌ NEGLIGIBLE

**Hypothesis:** Tuning RPA v3 `m_block_sizes` would close the 5.15× prefill TTFT gap (151ms vs H200's 29ms at L=4096).

**Method:** 23-configuration sweep on a dedicated debug pod with TPU access. Tested 16 `m_block_sizes` configs + 7 `p_block_sizes` (dedicated PREFILL launch) configs at prefill lengths {128, 512, 1024, 2048, 4096}.

**Result:** Block size tuning gives **1.15× kernel improvement** on a component that is **0.3% of TTFT**:

| Metric | Default | Best | Improvement |
|--------|---------|------|-------------|
| Raw kernel time @ L=4096 | 0.522ms | 0.455ms | 1.15× |
| End-to-end TTFT @ L=4096 | 151ms | ~151ms | ~0% |
| Kernel fraction of TTFT | 0.3% | 0.3% | — |

**Best config found:** `m_block_sizes=(512, 2048, 256, 512)` — committed as minor optimization.

**Root cause:** The RPA kernel is extremely fast (0.2–0.5ms). The 151ms TTFT is 99.7% dispatch/runtime overhead (XLA runtime, attention interface, Python scheduler). Block size tuning optimizes a negligible component.

### Phase 3: Expert Parallelism (EP=4) — ❌ OOM

**Hypothesis:** EP=4 would eliminate 2 MoE all-reduces per step (the dominant non-compile cost from P-4 trace).

**Result:** EP=4 with DP attention OOMs on 4-chip v7x:

| Config | HLO Temp Required | HBM Available | Gap |
|--------|------------------|---------------|-----|
| EP=4, max-model-len=8192 | 96.56G | 94.75G | -1.8% |
| EP=4, max-model-len=4096 (reduced) | 98.76G | 94.75G | -4.2% |

**Root cause:** DP attention requires each chip to hold all 4 KV heads' activations (vs 1 per chip in TP=4), increasing attention activation memory beyond the 94.75G HBM limit. Config reduction made it worse (smaller batch shapes changed compilation unfavorably). This is a fundamental constraint of the 4-chip topology, not a code bug.

**Code delivered (works in TP mode, ready for larger topology):**
- `cohere2_moe.py`: Conditional EP sharding (`expert_axis_name=EXPERT` when EP active, `None` in TP mode)
- Mutual-exclusion guard from deepseek_v3 pattern
- EP activates only with `NEW_MODEL_DESIGN=True, expert_parallelism=4, tensor_parallelism=1, enable_dp_attention=true`

---

## Final Performance Comparison (Config B Baseline)

| Metric | TPU v7x-4 | H200 (4× GPU) | Gap |
|--------|-----------|---------------|-----|
| Single-stream TPOT | 6.41ms (156 tok/s) | 3.95ms (253 tok/s) | 1.62× |
| Batch=16 output | 1,676 tok/s | 2,609 tok/s | 1.56× |
| Batch=32 output | 2,233 tok/s | 4,401 tok/s | 1.97× |
| Prefill TTFT @4096 | 151ms | 29ms | 5.15× |
| HBM BW utilization @c32 | 56.5% | — | — |

---

## Phase 4: async_scheduling (D2H Overlap) — ✅ SUCCESS

**Hypothesis:** Overlapping the previous step's D2H transfer with the next step's TPU forward dispatch would reduce the 5ms/step XLA runtime gap.

**Method:** Added `--async-scheduling` flag to vLLM args (no code change, no rebuild). Uses vLLM's `_pre_async_results` pattern (tpu_runner.py:1801-1849) — stashes step N's tokens via `jax.copy_to_host_async` (non-blocking), returns placeholder to scheduler, then at step N+1 materializes previous tokens and splices them into current input_ids on-TPU.

**Result:** First working optimization — significant throughput improvement:

| Concurrency | Config B output | async_scheduling output | Improvement | vs H200 gap |
|-------------|----------------|------------------------|-------------|-------------|
| c1 | 153 tok/s | 167 tok/s | +9% | 1.52× |
| c4 | 553 tok/s | 644 tok/s | +17% | — |
| c8 | 1,003 tok/s | 1,292 tok/s | **+29%** | **1.15×** |
| c16 | 1,676 tok/s | 1,967 tok/s | +17% | **1.33×** |
| c32 | 2,233 tok/s | 2,284 tok/s | +2% | 1.93× |

**Key finding:** At c8 (common serving batch size), the gap to H200 narrowed from 1.48× to **1.15×** — near parity. Correctness verified (identical output to Config B).

**Commit:** `c65ceb32` (perf branch), merged to base `b2c5bfd1`.

---

## Phase 5: Fused Graph Capture (single_step_decode) — ✅ CORRECTNESS VERIFIED + COMBINED WITH ASYNC = 2× THROUGHPUT

**Hypothesis:** Fusing the 4 separate per-step pjit dispatches (model_fn, _select_from_array, compute_logits, sample) into one jitted dispatch (no while_loop, unlike continue_decode) would reduce per-step XLA runtime overhead.

**Method:** New `single_step_decode` function in decode_loop.py + wiring in tpu_runner.py + precompile in compilation_manager.py + validation in tpu_platform.py. 25 structure tests pass.

**Root cause (CONFIRMED — three independent bugs, each causing garbled output):**

1. **Missing scheduler patch** (fixed in `b8e7bdaf`): `patch_vllm_scheduler_for_continue_decode()` was NOT called in the single_step_decode validation block (tpu_platform.py). Without the patch, `Scheduler._update_request_with_output` does not advance `num_computed_tokens` correctly for fused-decode outputs.

2. **Diagnostic delegation to unprecompiled continue_decode** (fixed in `cb70768e`): The diagnostic (commit `1f798d2c`) replaced the custom `_execute_single_step_decode` with a delegation to `_execute_continue_decode(max_steps=1)`. But when `enable_continue_decode=False`, `_precompile_continue_decode` does NOT run — so the delegated `continue_decode` was unprecompiled and compiled at runtime under `maybe_forbid_compile`.

3. **async_scheduling + single_step_decode incompatibility** (fixed in `9e989120` — THE blocking root cause): `_execute_single_step_decode` did NOT implement the async scheduling protocol (`_pre_async_results` / `_modify_prev_results` / `copy_to_host_async`). When `async_scheduling=True`, prefill sets `_pre_async_results`, decode step 1 consumes those (correct for first step), but decode step 2+ substitutes STALE prefill tokens every step → model receives wrong input → repetitive/garbled output.

**Fix:** Four commits — `b8e7bdaf` (scheduler patch) + `cb70768e` (restored custom wiring calling properly-precompiled `single_step_decode`) + `1e533f14` (forbid async+single_step combination) + `9e989120` (implement async protocol for single_step_decode — lifts the restriction).

The `9e989120` fix implements the full async protocol in `_execute_single_step_decode`:
- `_modify_prev_results()` — materializes previous step's tokens (replaces placeholders)
- `_update_placeholder()` — adds placeholders for current step
- `jax.copy_to_host_async(next_tokens)` — non-blocking D2H transfer
- Stashes in `_pre_async_results` for next-step token substitution
- Returns `AsyncTPUModelRunnerOutput`

**Validation (TPU v7x, async + single_step_decode, SKIP_JAX_PRECOMPILE=0):**

### Correctness: ✅ PASS
Coherent code generation on all test prompts (`def hello_world():`, `def fibonacci(n):`, `import json`). No recompilation errors. The async protocol implementation works perfectly — no stale token substitution.

### Performance — Combined async + single_step_decode

| Concurrency | Config B | async only | single_step only | **COMBINED** | **vs Config B** | **vs H200** | **Gap** |
|---|---|---|---|---|---|---|---|
| c1  | 153   | 167   | 97    | **168**   | +9.5%  | 249   | 1.49× |
| c4  | 553   | 644   | 378   | **650**   | +17.5% | 850   | 1.31× |
| c8  | 1,003 | 1,292 | 761   | **1,292** | +28.8% | 1,486 | **1.15×** |
| c16 | 1,676 | 1,967 | 1,614 | **2,884** | +72.1% | 2,609 | **0.90× — TPU WINS!** |
| c32 | 2,233 | 2,284 | 2,687 | **4,499** | +101.5%| 4,401 | **0.98× — TPU WINS!** |

### Key findings:
1. **Low batch fully recovered** (c1: 168 vs single_step-only's 97) — async overlap eliminates the 4ms/step fused dispatch overhead that hurt single_step alone.
2. **High batch stacks both optimizations** — c32: 4,499 tok/s = 2× Config B, 2× async-only, 1.7× single_step-only.
3. **c8 matches async-only** (1,292) — no regression anywhere.
4. **TPU BEATS H200 at c16+** — the TPU's 1.54× bandwidth and 2.18× compute advantage is finally realized.
5. TTFT stable at all concurrency levels (1421-1826ms) — no degradation at high batch.

**Status:** ✅ PRODUCTION CONFIG. Combined async_scheduling + single_step_decode is the best configuration across all batch sizes. Build `4e4a619a`, image `gcr.io/northam-ce-mlai-tpu/nmc-inference:latest`.

---

## Updated Performance Comparison (with combined async + single_step_decode)

| Metric | Config B | async only | **COMBINED** | H200 | Gap (combined vs H200) |
|--------|---------|------------|-------------|------|------------------------|
| Single-stream TPOT | 6.41ms | ~6.0ms | ~5.9ms | 3.95ms | 1.49× |
| c8 output throughput | 1,003 tok/s | 1,292 tok/s | **1,292 tok/s** | 1,486 tok/s | **1.15×** |
| c16 output throughput | 1,676 tok/s | 1,967 tok/s | **2,884 tok/s** | 2,609 tok/s | **0.90× — TPU WINS!** |
| c32 output throughput | 2,233 tok/s | 2,284 tok/s | **4,499 tok/s** | 4,401 tok/s | **0.98× — TPU WINS!** |
| Prefill TTFT @4096 | 151ms | 102ms | 103ms | 29ms | 3.55× |

---

## Root Cause Analysis

The TPU v7x has superior hardware specs vs 4× H200:
- **1.54× more HBM bandwidth** (29.6 vs 19.2 TB/s)
- **2.18× more BF16 compute** (8,628 vs 3,956 TFLOP/s)

Despite this, the TPU is 1.5–2× slower. The bottleneck is **XLA runtime dispatch overhead**:

1. **Per-step dispatch (~5ms):** Each decode step fires multiple separate pjit dispatches (model_fn, compute_logits, sample, logprobs). XLA runtime scheduling adds ~5ms on top of <1ms actual TPU execution. This is the dominant cost at all batch sizes.

2. **Prefill dispatch amplification:** The 5ms/step overhead multiplies across prefill steps. At L=4096, this produces ~150ms TTFT (vs the kernel's 0.5ms). The H200's CUDA runtime has lower per-step dispatch overhead.

3. **MoE all-reduce (2 collectives/step):** Costs ~10ms from P-4 trace. EP would eliminate this but OOMs on 4 chips.

4. **TPU duty cycle:** 9.16% at batch=3 (P-4 trace) — 91% of time is spent in host-side dispatch, not TPU execution.

---

## What Would Close the Gap

These levers were identified but not actionable on the current 4-chip topology:

| Lever | Expected impact | Status |
|-------|----------------|--------|
| **async_scheduling** | Overlaps D2H with next step input prep | ✅ **DONE** — +8-29% throughput |
| **single_step_decode (fused dispatch)** | Fuse 4 dispatches into 1 | ✅ **DONE** — +20% at c32 alone |
| **async + single_step combined** | Stack both optimizations | ✅ **DONE** — **2× Config B at c32, TPU beats H200 at c16+** |
| **Pathways (JAX_PLATFORMS=proxy)** | True async/remote dispatch — eliminates per-step sync | Future feature, not yet production-ready |
| **EP on larger topology (8+ chips)** | Eliminates 2 all-reduces/step | OOMs on 4 chips (DP attention memory) |
| **CUDA-graph-style capture for prefill** | Eliminate per-step dispatch during prefill | Not implemented in tpu-inference |

---

## Optimizations Applied

| Change | Impact | Commit |
|--------|--------|--------|
| **async_scheduling + single_step_decode (COMBINED)** | **2× throughput over Config B at c32. TPU beats H200 at c16+. Production config.** | `9e989120` + `4e4a619a` (build) |
| async_scheduling (alone) | +8-29% throughput (gap to H200: 1.15× at c8) | `b2c5bfd1` |
| single_step_decode async protocol | Implements _pre_async_results/_modify_prev_results/copy_to_host_async in _execute_single_step_decode | `9e989120` |
| `m_block_sizes=(512, 2048, 256, 512)` | 1.15× kernel improvement (0.3% of TTFT) | `1f02c187` |
| EP code infrastructure | Ready for larger topology | `5e07e0c5` |
| continue_decode fix | Correctness fix (works, just slower) | `4a0a6fc4` |
| Phase 2 code infrastructure | `m_block_sizes`/`p_block_sizes`/`chunk_prefill_size` fields | `3b6b81e4` |

---

## Conclusion

The TPU v7x port of North Mini Code is **functionally complete and serving with the combined async_scheduling + single_step_decode configuration** — the best-performing config across all batch sizes. The model runs end-to-end with correct output at **4,499 tok/s decode throughput (batch=32, 2× over Config B baseline)**.

**The TPU now BEATS H200 at c16 and c32** — the milestone we've been working toward. The TPU's 1.54× more bandwidth and 2.18× more compute than 4× H200 is finally realized at high batch through the combination of:
1. **Fused dispatch** (single_step_decode): 1 jitted dispatch instead of 4, eliminating per-step XLA dispatch overhead
2. **Async D2H overlap** (async_scheduling): overlaps token transfer with next step's compute

The gap went from 1.5–2.5× (Config B) down to **parity/surpassing at c16+** (0.90× at c16, 0.98× at c32). The remaining gap at low batch (c1: 1.49×) and prefill (3.55× at 4096 tokens) is XLA runtime overhead that would require Pathways (async remote dispatch) or CUDA-graph-style prefill capture to close further.

**Optimization journey:**
- Config B (batch tuning): 2.25× over baseline ✅
- async_scheduling: +8-29% at c4-c16 ✅
- single_step_decode alone: +20% at c32, -37% at c1 (high-batch-only) ✅
- **async + single_step combined: 2× Config B at c32, TPU beats H200 at c16+** ✅ **PRODUCTION**
- continue_decode: slower (while_loop overhead) ❌
- EP=4: OOM (DP attention memory on 4 chips) ❌
- Prefill block sizes: negligible (0.3% of TTFT) ❌
