# TPU Porting Task List: North Mini Code (Cohere2 MoE)

> Tracking the port of `CohereLabs/North-Mini-Code-1.0` (`cohere2_moe` / `Cohere2MoeForCausalLM`)
> into `tpu-inference` on the `feature/north-mini-code` branch.
> Key finding: **no brand-new Pallas kernels required** — needed kernels (fused_moe sigmoid routing,
> RPA v3 sliding-window, interleaved RoPE) already exist. Work = model integration + weight remapping
> + wiring verification + v7x tuning. See `porting_backlog.md` / `PORTING_OVERVIEW.md`.

## Orchestrator Setup (COMPLETE)
- [x] Initialize accelerator-agents submodule
- [x] Clone chiefkarlin/tpu-inference fork; set up feature/north-mini-code base branch
- [x] Code-verify porting_backlog.md against actual repo (6 discrepancies found & documented in §8)
- [x] Push corrected docs (§8 findings) to fork feature/north-mini-code
- [x] Create + push feature/north-mini-code-kernels and feature/north-mini-code-perf branches
- [x] Spawn tpu-kernel-engineer + tpu-performance-engineer agents (Hub mode)
- [x] Send bootstrap instructions (clone fork, checkout branch, init submodule, read corrected docs)

## Collaboration Model (Hub mode: each agent has its own container/clone)
- Base branch `feature/north-mini-code` on `chiefkarlin/tpu-inference` fork — holds all docs + shared work.
- Kernel Engineer works on branch `feature/north-mini-code-kernels` (off base); pushes to fork.
- Performance Engineer works on branch `feature/north-mini-code-perf` (off base); pushes to fork.
- Orchestrator reviews + merges agent branches back into `feature/north-mini-code`; resolves conflicts.
- Agents coordinate through the Orchestrator via `scion message`.

## Phase 1: Model Integration (Kernel Engineer — K-1…K-6) — COMPLETE ✅
- [x] K-1 Sliding-window decode wiring — custom `cohere2_attention.py` (regular RPA v3, head_dim=128, d_block_sizes=(1,4096,1,2048))
- [x] K-2 MoE topology verified (128 exp, 768 interm, topk=8, sigmoid, no-renorm via JaxMoE delegation)
- [x] K-3 `models/jax/cohere2_moe.py` → `Cohere2MoeForCausalLM` (parallel block, hybrid schedule, conditional interleaved RoPE, tied head) — commit `1fc5507b` + fixes `be319e71`
- [x] K-4 Weight remapping via JaxMoE._load_weights (128 per-expert → fused EDF/EFD; dense layer 0 prefix_dense_intermediate_size)
- [x] K-5 Registered `Cohere2MoeForCausalLM` in `models/common/model_loader.py`
- [x] K-6 28 AST structural tests — all PASS (independently verified by orchestrator + perf eng)
- [x] Code review: tri-directional review found 3 bugs → 3 fixes applied → re-reviewed → 28/28 tests PASS
- [x] Hand off to Performance Engineer — merged into perf branch `65b98ebf`

## Phase 2: v7x Bring-up & Tuning (Performance Engineer — P-1…P-4)
- [x] P-1 MoE tuning inspection — **CORRECTED**: default path is GMM_TP→gmm_v2 (calculate_tiling heuristic, no tuned table). All tuned_block_sizes.py files are OFF the default path. Decision: accept heuristic for baseline. See backlog §9.
- [x] P-2 RPA v3 tuning inspection — **CORRECTED**: RPA v3 clamps KV to sliding_window internally. Real lever = explicit d_block_sizes=(1,4096,1,2048) in custom attention module (adopted by kernel eng). See backlog §9.
- [!] P-3 GKE auth → build+push Docker to GAR → inject HF_TOKEN → kubectl run on v7x-4 / single-host-vllm — **BLOCKED on cluster creds** (SOLE remaining blocker; escalated to user). Docker build script + Cloud Build config + k8s manifest all ready on perf branch.
- [~] P-4 Capture ML Diagnostics trace + per-kernel timings; identify HBM bottlenecks — profiling plan (307 lines) + trace analysis script ready; needs cluster access.

## Phase 3: Integration & Validation (Orchestrator)
- [ ] Review & merge `feature/north-mini-code-kernels` → `feature/north-mini-code`
- [ ] Review & merge `feature/north-mini-code-perf` → `feature/north-mini-code`
- [ ] End-to-end NMC inference on v7x-4 with real weights (correctness sanity) — blocked on P-3
- [ ] No regression on existing registered models

## Status Legend
- `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked
