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

## Phase 1: Model Integration (Kernel Engineer — K-1…K-6) — IN PROGRESS
- [~] K-1 Verify sliding-window reaches RPA v3 kernel in **decode**; wire if dropped (HIGH, correctness risk)
      → Root cause CONFIRMED: base Attention drops sliding_window in decode; solution = custom attention module
        modeled on gpt_oss_attention.py but using regular ragged_paged_attention (head_dim=128, not hd64)
- [ ] K-2 Verify fused_moe/megablox GMM accepts NMC topology (128 exp, 768 interm, topk=8, sigmoid, no-renorm)
- [ ] K-3 Implement `models/jax/cohere2_moe.py` → `Cohere2MoeForCausalLM` (parallel block, hybrid schedule, conditional interleaved RoPE, tied head)
- [ ] K-4 Weight remapper (128 per-expert → fused 3-D; dense layer 0; tied head)
- [ ] K-5 Register `Cohere2MoeForCausalLM` in `models/common/model_loader.py`
- [ ] K-6 Unit tests (weight shapes, sigmoid-no-renorm routing, parallel-block forward, RoPE conditional)
- [ ] Hand off compilable model to Performance Engineer

## Phase 2: v7x Bring-up & Tuning (Performance Engineer — P-1…P-4) — IN PROGRESS
- [x] P-1 MoE tuning inspection — **CORRECTED**: default path is GMM_TP→gmm_v2 (calculate_tiling heuristic, no tuned table). All tuned_block_sizes.py files are OFF the default path. Decision: accept heuristic for baseline; revisit tile_info wiring post-baseline if profiling proves bottleneck. See backlog §9.
- [x] P-2 RPA v3 tuning inspection — **CORRECTED**: RPA v3 clamps KV to sliding_window internally (no over-fetch). tuned_block_sizes.py is dead code for regular path. Real lever = explicit block_sizes in custom attention module (kernel eng owns). Recommendation: bkv_csz=2048-4096. See backlog §9.
- [!] P-3 GKE auth → build+push Docker to GAR → inject HF_TOKEN → kubectl run on v7x-4 / single-host-vllm — **BLOCKED on cluster creds** (metadata concealment; escalated to user). Perf eng proceeding with P-4 prep + Docker scaffolding + kubectl manifests in the meantime.
- [~] P-4 Capture ML Diagnostics trace + per-kernel timings; identify HBM bottlenecks — profiling plan being drafted (no cluster access needed for plan)

## Phase 3: Integration & Validation (Orchestrator)
- [ ] Review & merge `feature/north-mini-code-kernels` → `feature/north-mini-code`
- [ ] Review & merge `feature/north-mini-code-perf` → `feature/north-mini-code`
- [ ] End-to-end NMC inference on v7x-4 with real weights (correctness sanity)
- [ ] No regression on existing registered models

## Status Legend
- `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked
