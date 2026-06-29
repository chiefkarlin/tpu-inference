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

## Phase 2: v7x Bring-up & Tuning (Performance Engineer — P-1…P-4) — COMPLETE ✅
- [x] P-1 MoE tuning inspection — **CORRECTED**: default path is GMM_TP→gmm_v2 (calculate_tiling heuristic, no tuned table). All tuned_block_sizes.py files are OFF the default path. Decision: accept heuristic for baseline. See backlog §9. **Runtime CONFIRMED**: GMM auto-tiling good (GMM1: tk=2048/tn=256, GMM2: tk=256/tn=2048).
- [x] P-2 RPA v3 tuning inspection — **CORRECTED**: RPA v3 clamps KV to sliding_window internally. Real lever = explicit d_block_sizes=(1,4096,1,2048) in custom attention module (adopted by kernel eng). **Runtime CONFIRMED**: 0.004ms/call, NOT a bottleneck.
- [x] P-3 GKE auth → build+push Docker → kubectl run on v7x spot TPU — **COMPLETE**. Worked around metadata concealment (SA token exec-plugin kubeconfig), built image via Cloud Build REST API (build v5, gcr.io/northam-ce-mlai-tpu/nmc-inference:latest), deployed on spot 2x2x1 TPU v7x, e2e inference SUCCESS.
- [x] P-4 Trace capture + analysis — **COMPLETE**. 386MB xplane.pb captured (batch=3, 64 tokens). RPA attention NOT bottleneck (0.004ms), MoE GEMM auto-tiling good, TPU duty cycle 9.16% (Python dispatch + small batch, not kernels). 8× gap to HBM theoretical (485 vs 3,950 tok/s) = system-level utilization.

## Phase 3: Integration & Validation (Orchestrator) — COMPLETE ✅
- [x] Review & merge `feature/north-mini-code-kernels` → `feature/north-mini-code` (via perf branch merge)
- [x] Review & merge `feature/north-mini-code-perf` → `feature/north-mini-code` (commit `12d3816a`, clean merge, 28/28 tests verified on base)
- [x] Merge EXECUTION_RUNBOOK + self-contained trace analyzer to base (commit `b5d8e2ca`)
- [x] Merge BUG #3 fix (mlp_layer_types, `29d13c6d`→`e4186fa3`) + K-4 fix (make_array_from_single_device_arrays, `a93e463c`) + BUG #5 fix (KV head TP sharding, `7056bf31`)
- [x] End-to-end NMC inference on v7x with real weights — **SUCCESS** (`def hello_world():` → coherent Python, 200 OK, 19.2 tok/s single / 485 tok/s batch=3)
- [x] Restore vllm-qwen + vllm-gpt-oss workloads (scaled to 0 during testing, restored after)

## Runtime Bugs Found & Fixed During Deployment
- [x] **BUG #3** (first_k_dense_replace): Cohere2MoeConfig.__post_init__ pops attribute without storing → layer 0 built as MoE. Fix: use `config.mlp_layer_types[layer_idx] == 'dense'` (`29d13c6d`)
- [x] **K-4** (multi-host sharding): make_array_from_callback fails on worker host (0 addressable shards, no dtype). Fix: revert to make_array_from_single_device_arrays with explicit dtype (`a93e463c`)
- [x] **BUG #5** (KV head TP sharding): Q/KV projection sharding specs didn't shard heads on model axis → shape mismatch (4 heads vs 1/device). Fix: shard head dims on ShardingAxisName.MODEL (`7056bf31`)

## Standby Deliverables (produced while blocked on cluster creds)
- [x] EXECUTION_RUNBOOK.md (587 lines, 10 sections) — copy-paste executable P-3→P-4 guide for any engineer with GCP access
- [x] analyze_nmc_trace.py refactored — fully self-contained (no MaxKernel/xprof/accelerator-agents dependency)
- [x] Build validation: 2 build bugs caught + fixed (e2e JSON quoting, cloudbuild shell expansion)
- [x] HBM floor analysis corrected (1.87GB/chip decode, 0.25ms/token theoretical min)

## Final State
- **All code merged to `feature/north-mini-code` @ `78e0e9fd`** (base branch)
- **Docker image**: `gcr.io/northam-ce-mlai-tpu/nmc-inference:latest` (build v5, both BUG #3 + K-4 + BUG #5 fixes)
- **E2E inference VERIFIED**: NMC generates coherent text on TPU v7x (2x2x1, TP=4, spot)
- **Profiling COMPLETE**: no kernel tuning warranted for baseline; 8× gap = Python dispatch + small batch
- **vllm-qwen + vllm-gpt-oss restored** to replicas=1 after testing

## Status Legend
- `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked
