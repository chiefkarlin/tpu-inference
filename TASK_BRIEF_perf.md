# TASK BRIEF — Performance Engineer: North Mini Code v7x bring-up & tuning

**Worktree:** `/workspace/nmc-perf` (branch `feature/north-mini-code-perf`, off `feature/north-mini-code`)
**You are:** tpu-performance-engineer. **Repo:** fork of `vllm-project/tpu-inference`.
**Full backlog:** see `/workspace/tpu-inference/PORTING_OVERVIEW.md` and
`/workspace/experiments/north-mini-code/porting_backlog.md` (read both first).

## Objective
Bring up end-to-end inference of **CohereLabs/North-Mini-Code-1.0** on the **robv-tpu-v7x** GKE cluster
and tune the existing kernels for NMC's shapes. The model integration is being done in parallel by the
Kernel Engineer on `feature/north-mini-code-kernels`; coordinate via the Orchestrator. You own
**P-1 … P-4**.

## Tools available
- `/workspace/accelerator-agents/JAXBench/` — benchmarking harness.
- `/workspace/accelerator-agents/MaxKernel/` — profiling / Pallas tuning help.

## Cluster & infra (from experiments/north-mini-code/README.md)
- **GKE cluster:** `gke_northam-ce-mlai-tpu_us-central1-c_robv-tpu-v7x`. Auth:
  `gcloud container clusters get-credentials gke_northam-ce-mlai-tpu_us-central1-c_robv-tpu-v7x --region us-central1-c`.
- **Topology:** single-node `v7x-4`; node pool `single-host-vllm` (node type `tpu7x-standard-4t`).
- **Docker:** build a custom image using the repo `docker/` Dockerfile as reference; push to GAR;
  submit via `kubectl`. Use the repo's existing Dockerfile/scripts as the starting point.
- **Weights:** download `CohereLabs/North-Mini-Code-1.0` from HuggingFace at runtime by injecting
  `HF_TOKEN` (available in the Scion project environment) into the pod/job. The bf16 variant is ~61GB
  (49 safetensors shards).
- Confirm cluster access first (`kubectl get nodes`) before building images.

## NMC shapes to tune for (v7x-4)
- **MoE grouped GEMM:** 128 experts, 8 active/token, intermediate=768, hidden=2048, gated SiLU,
  sigmoid routing. Small per-expert intermediate (768) — SRAM block sizing matters.
- **Attention (RPA v3):** GQA 32 q-heads / 4 kv-heads, head_dim=128, sliding_window=4096 (sliding
  layers) + full attention (full layers). Hybrid schedule.
- **dtype:** bfloat16.

## Tasks (P-1 … P-4)
- **P-1 (HIGH):** Tune fused_moe / megablox block sizes for NMC on v7x. Inspect
  `kernels/fused_moe/v1/tuned_block_sizes.py` and `kernels/megablox/tuned_block_sizes.py`; add/verify
  entries for `(E=128, I=768, D=2048, topk=8)`.
- **P-2 (HIGH):** Tune RPA v3 block sizes for `(head_dim=128, sliding_window=4096, GQA 32:4)` on v7x.
  Inspect `kernels/ragged_paged_attention/v3/tuned_block_sizes.py` (and `..._hd64.py` if applicable).
- **P-3 (HIGH):** v7x GKE auth → build+push Docker image to GAR → inject `HF_TOKEN` → kubectl run on
  `v7x-4` / `single-host-vllm`. Get a first end-to-end NMC inference run (short prompt) working.
- **P-4 (MED):** Capture ML Diagnostics trace; report per-kernel timings (prefill + decode); identify
  HBM-bound bottlenecks (esp. MoE GEMM, sliding-window attention). Track against a latency target
  (confirm target with Orchestrator if none provided).

## Dependencies & sequencing
- **P-3 (end-to-end run) depends on the Kernel Engineer landing a compilable `Cohere2MoeForCausalLM`.**
  Start P-1/P-2 (block-size inspection/tuning can begin against the existing kernels) and P-3 cluster
  auth + Docker scaffolding NOW, in parallel. Pull the model file from
  `feature/north-mini-code-kernels` (via the Orchestrator) once ready.
- If the Kernel Engineer's sliding-window decode wiring (K-1) changes attention behavior, re-profile.

## Constraints
- Do not modify kernel correctness logic without coordinating with the Kernel Engineer (tuning =
  block-size/table changes only).
- Keep all work on `feature/north-mini-code-perf`; commit incrementally; message the Orchestrator via
  `scion` on milestones/blockers.

## Done criteria
- NMC runs end-to-end on v7x-4 with real weights (correctness sanity check on a short prompt).
- Tuned block-size tables committed for NMC shapes.
- Profiling trace + per-kernel timing report delivered to the Orchestrator.
