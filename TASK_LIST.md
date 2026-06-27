# TPU Porting Task List: North Mini Code (Cohere2 MoE)

> Tracking the port of `CohereLabs/North-Mini-Code-1.0` (`cohere2_moe` / `Cohere2MoeForCausalLM`)
> into `tpu-inference` on the `feature/north-mini-code` branch.
> Key finding: **no brand-new Pallas kernels required** — needed kernels (fused_moe sigmoid routing,
> RPA v3 sliding-window, interleaved RoPE) already exist. Work = model integration + weight remapping
> + wiring verification + v7x tuning. See `porting_backlog.md` / `PORTING_OVERVIEW.md`.

## Worktrees
- `/workspace/tpu-inference` — primary repo, branch `feature/north-mini-code` (Orchestrator).
- `/workspace/nmc-kernels` — Kernel Engineer, branch `feature/north-mini-code-kernels`.
- `/workspace/nmc-perf` — Performance Engineer, branch `feature/north-mini-code-perf`.

## Phase 1: Model Integration (Kernel Engineer — K-1…K-6)
- [ ] K-1 Verify sliding-window reaches RPA v3 kernel in **decode**; wire if dropped (HIGH, correctness risk)
- [ ] K-2 Verify fused_moe/megablox GMM accepts NMC topology (128 exp, 768 interm, topk=8, sigmoid, no-renorm)
- [ ] K-3 Implement `models/jax/cohere2_moe.py` → `Cohere2MoeForCausalLM` (parallel block, hybrid schedule, conditional interleaved RoPE, tied head)
- [ ] K-4 Weight remapper (128 per-expert → fused 3-D; dense layer 0; tied head)
- [ ] K-5 Register `Cohere2MoeForCausalLM` in `models/common/model_loader.py`
- [ ] K-6 Unit tests (weight shapes, sigmoid-no-renorm routing, parallel-block forward, RoPE conditional)
- [ ] Hand off compilable model to Performance Engineer

## Phase 2: v7x Bring-up & Tuning (Performance Engineer — P-1…P-4)
- [ ] P-1 Tune fused_moe/megablox block sizes for (E=128, I=768, D=2048, topk=8) on v7x
- [ ] P-2 Tune RPA v3 block sizes for (head_dim=128, sliding_window=4096, GQA 32:4) on v7x
- [ ] P-3 GKE auth → build+push Docker to GAR → inject HF_TOKEN → kubectl run on v7x-4 / single-host-vllm
- [ ] P-4 Capture ML Diagnostics trace + per-kernel timings; identify HBM bottlenecks

## Phase 3: Integration & Validation (Orchestrator)
- [ ] Review & merge `feature/north-mini-code-kernels` → `feature/north-mini-code`
- [ ] Review & merge `feature/north-mini-code-perf` → `feature/north-mini-code`
- [ ] End-to-end NMC inference on v7x-4 with real weights (correctness sanity)
- [ ] No regression on existing registered models

## Status Legend
- `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked
