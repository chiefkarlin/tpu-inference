# TASK BRIEF — Kernel Engineer: North Mini Code (Cohere2 MoE) port

**Worktree:** `/workspace/nmc-kernels` (branch `feature/north-mini-code-kernels`, off `feature/north-mini-code`)
**You are:** tpu-kernel-engineer. **Repo:** fork of `vllm-project/tpu-inference`.
**Full backlog:** see `/workspace/tpu-inference/PORTING_OVERVIEW.md` and
`/workspace/experiments/north-mini-code/porting_backlog.md` (read both first).

## Objective
Implement a JAX-native (`flax_nnx`) model for **CohereLabs/North-Mini-Code-1.0**
(`model_type=cohere2_moe`, arch `Cohere2MoeForCausalLM`) so it loads and runs in tpu-inference.
**No brand-new Pallas kernels are required from scratch** — the needed kernels already exist. Your job
is **model integration + weight remapping + wiring verification + unit tests** (items K-1…K-6 below).

## ⚠️ VERIFICATION CORRECTIONS (code-confirmed by Orchestrator — READ FIRST, supersedes anything below)
The backlog/brief were spot-checked against the actual repo on `feature/north-mini-code`. Key corrections:
1. **K-1 sliding-window decode (root cause + solution):** The base `Attention.attention()` decode path
   (`layers/jax/attention/attention.py` ~L243-250) calls `ragged_paged_attention` **WITHOUT**
   `sliding_window` → setting `attention_metadata.sliding_window` alone is **silently dropped in decode**.
   Prefill is fine (`layers/common/attention_interface.py:405` forwards `sliding_window=attention_chunk_size`).
   **Solution template:** `layers/jax/attention/gpt_oss_attention.py` (NOT `models/jax/`) — a custom
   attention module whose `attention()` calls the RPA kernel with `sliding_window=md.sliding_window` (L195).
   gpt_oss uses `ragged_paged_attention_hd64` (its head_dim==64). **NMC head_dim==128 → `use_hd64=False`
   (attention_interface.py:383) → use the regular `ragged_paged_attention` (v3), NOT hd64.** Build a custom
   attention module for NMC on this pattern; set `attention_metadata.sliding_window=4096|None` per layer
   (gpt_oss.py:537-538) + `attention_chunk_size=sliding_window` for prefill (gemma4.py:525).
2. **MoE API names:** `layers/common/fused_moe_gmm.py::fused_moe_func` uses `scoring_fn`+`activation`
   (NOT `router_act`); `renormalize: bool` is **required, no default**. `router_act="sigmoid"` is on the
   `Router` class (llama4.py:524). deepseek_v3 uses `scoring_func` (default `"sigmoid"`) + `norm_topk_prob`
   (set `False` for NMC → skip the renorm branch at deepseek_v3.py:1110-1119).
3. **Model interface (K-3):** constructor MUST be `__init__(self, vllm_config, rng_key, mesh)` (3 args) —
   `_get_nnx_model` (model_loader.py:144) calls `model_class(vllm_config, rng, mesh)`. The validator only
   checks for a `vllm_config` kwarg, so `__init__(self, vllm_config)` alone would pass validation but
   **crash at construction**. `__call__(self, kv_caches, input_ids, attention_metadata, ...)`.
4. **Weight stacking (K-4):** do NOT copy deepseek_v3's `load_weights` (delegates to `JaxAutoWeightsLoader`).
   Template: `models/jax/gemma4.py:216-249` + `models/jax/gpt_oss.py:283-294,358-374` (EFD/EDF layouts,
   `permute_dims=(0,2,1)`). NMC stores 128 **separate** per-expert matrices → stack to
   `(128, 2*768, 2048)` / `(128, 2048, 768)`.
5. **RoPE:** `apply_rope` defaults `rope_input_ordering="split"`; explicitly pass `"interleaved"` (NeoX
   adjacent-pair). Conditional per-layer via `use_attention_rope` (attention.py:110, llama4.py:518).
6. **No single reference combines** `attention_chunk_size=sliding_window` + `rope_input_ordering="interleaved"`
   + conditional `use_attention_rope`. llama4 has interleaved+conditional-RoPE but uses a fixed
   `attention_chunk_size=None/8192`; gemma4 has `attention_chunk_size=sliding_window` but split-RoPE.
   Combine deliberately for NMC.
7. **`layer_types` strings** in config.json are `"full_attention"` / `"sliding_attention"` (not
   `"full"`/`"sliding"`). Full §8 verification section is in `porting_backlog.md`.

## Tools available
- `/workspace/accelerator-agents/MaxCode/` — PyTorch→JAX/MaxText conversion guidance.
- `/workspace/accelerator-agents/MaxKernel/` — Pallas kernel writing/profiling/test-harness help.
- You can run CPU/local JAX unit tests here; **TPU execution on v7x is the Performance Engineer's job.**

## Closest existing references (READ THESE)
- `tpu_inference/models/jax/deepseek_v3.py` — sigmoid routing + `norm_topk_prob` handling; MoE layer
  structure; weight loading patterns. **Your primary structural template.**
- `tpu_inference/models/jax/gpt_oss.py` — per-layer full/sliding attention schedule
  (`attention_metadata.sliding_window` set per layer, L537-538); hybrid layer construction.
- `tpu_inference/models/jax/gemma4.py` & `llama4.py` — sliding window via `attention_chunk_size`,
  interleaved RoPE via `rope_input_ordering="interleaved"`, conditional RoPE via `use_attention_rope`.
- `tpu_inference/layers/jax/attention/attention.py` — base `Attention` module API.
- `tpu_inference/layers/jax/rope_interface.py::apply_rope` — supports `rope_input_ordering="interleaved"`.
- `tpu_inference/layers/jax/moe/moe.py`, `tpu_inference/layers/common/fused_moe_gmm.py`,
  `tpu_inference/kernels/fused_moe/v1/kernel.py`, `tpu_inference/kernels/megablox/gmm*.py` — MoE path.
- `tpu_inference/models/common/model_loader.py` — registry + `register_model`; model interface contract
  (`_validate_model_interface`: `__init__(vllm_config)` + `__call__(kv_caches, input_ids, attention_metadata)`).

## Architecture you must reproduce (EXACTLY)
- **49 layers.** `layer_types = [full, sliding, sliding, sliding]` repeated ×12 (=48) + 1 final `full`.
- **Layer 0 = DENSE** MLP (`first_k_dense_replace=1`, `intermediate_size=3072`).
- **Layers 1–48 = MoE:** 128 experts, `num_experts_per_tok=8`, `intermediate_size=768` per expert,
  `num_shared_experts=0` (NO shared experts). Gated SiLU: `down(silu(gate)*up)*weight`.
- **MoE routing:** `expert_selection_fn=sigmoid`, `norm_topk_prob=false`. Router = `mlp.gate` (128×2048).
  top-k on logits → `sigmoid(selected)` → **NO renormalization** (raw sigmoid scores are expert weights).
  Use `router_act="sigmoid"`, `renormalize=False` on the existing MoE path.
- **Attention:** GQA 32 q-heads / 4 kv-heads, `head_dim=128`, no bias, `use_qk_norm=false`.
- **Sliding window = 4096** on `sliding` layers; `full` layers attend to all KV (sliding_window=None).
- **RoPE:** interleaved (NeoX), `rope_theta=50000`, `rope_scaling=None`. **CRITICAL RULE — RoPE applied
  iff `sliding_window is not None OR force_rope`**, where
  `force_rope = (layer is dense) and (prefix_dense_sliding_window_pattern==1)`:
  - sliding layers → RoPE ON (interleaved)
  - dense layer 0 → RoPE ON (force_rope)
  - **full-attention MoE layers → RoPE OFF**
  Implement via per-layer `use_attention_rope` + `rope_input_ordering="interleaved"`.
- **PARALLEL BLOCK (new — no JAX precedent):** each decoder layer uses a SINGLE shared `input_layernorm`
  (T5-style RMSNorm, variance only, no centering, eps=1e-6) feeding both attention and MLP, and NO
  `post_attention_layernorm`:
  `hidden = x + attn(input_layernorm(x)) + mlp(input_layernorm(x))`
  (All existing models use sequential blocks with two norms — do NOT copy that; implement the parallel
  form explicitly.)
- **Embeddings:** `vocab_size=262144`, `hidden_size=2048`. `lm_head` TIED to `embed_tokens`.
  `logit_scale=1.0`. Final RMSNorm before head.
- **dtype:** bfloat16.

## Weight naming (safetensors — must remap in `load_weights`)
- `model.embed_tokens.weight` (262144×2048) → tied `lm_head` (no separate lm_head key).
- `model.norm.weight` → final RMSNorm.
- Layer 0 (dense): `model.layers.0.{input_layernorm.weight, mlp.{gate,up,down}_proj.weight,
  self_attn.{q,k,v,o}_proj.weight}`.
- Layers 1–48 (MoE): `model.layers.{i}.input_layernorm.weight`;
  `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`; `model.layers.{i}.mlp.gate.weight` (router);
  `model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight` for e in 0..127 — **128 SEPARATE
  per-expert matrices** (gate/up=(768,2048), down=(2048,768)).
  → **Stack** per-expert `gate`+`up` into fused `(128, 2*768, 2048)` and `down` into `(128, 2048, 768)`
  to match the grouped-GEMM layout the megablox/fused_moe kernels consume. Confirm the exact expected
  layout against `fused_moe_gmm.py`/`gmm*.py` before finalizing.

## Tasks (K-1 … K-6)
- **K-1 (HIGH, correctness risk):** Verify sliding-window reaches the RPA v3 kernel in **decode**.
  The base `Attention.attention()` (`layers/jax/attention/attention.py`) calls `ragged_paged_attention`
  WITHOUT forwarding `sliding_window`. Determine whether `attention_metadata.sliding_window` (gpt_oss
  pattern) or `attention_chunk_size` (gemma4 pattern) reaches the kernel in decode; if not, wire it so
  NMC's sliding layers get `sliding_window=4096` and full layers get `None`. Document your finding.
- **K-2 (HIGH):** Verify fused_moe/megablox GMM accepts NMC topology (128 experts, 768 interm, topk=8,
  sigmoid, no-renorm) end-to-end; confirm the fused 3-D weight layout matches your loader output.
- **K-3 (HIGH, bulk):** Implement `tpu_inference/models/jax/cohere2_moe.py` → `Cohere2MoeForCausalLM`
  with the parallel block, hybrid schedule, conditional interleaved RoPE, tied head, logit_scale.
  Must satisfy `_validate_model_interface` (`__init__(vllm_config, rng, mesh)`,
  `__call__(kv_caches, input_ids, attention_metadata, ...)`, `compute_logits`, `load_weights`).
- **K-4 (HIGH):** Weight remapper as described above (+ dense layer 0, tied head).
- **K-5 (MED):** Register in `models/common/model_loader.py::_get_model_architecture`
  (`_MODEL_REGISTRY["Cohere2MoeForCausalLM"] = Cohere2MoeForCausalLM`). Keep `flax_nnx` path (do NOT add
  to `_VLLM_PREFERRED_ARCHITECTURES`).
- **K-6 (MED):** Unit tests — weight shapes; routing (sigmoid, no-renorm) vs tiny reference; parallel-block
  forward vs reference; RoPE interleaved + conditional correctness. Use MaxKernel test-harness help.

## Constraints
- Match the RoPE conditional rule EXACTLY (off-by-one here = silent correctness bug).
- Preserve sigmoid-no-renorm routing exactly.
- Do NOT break any existing registered model (run any existing model tests you can).
- Commit incrementally to `feature/north-mini-code-kernels`; message the Orchestrator via `scion` when
  ready for review/merge, or if blocked.

## Done criteria
A `Cohere2MoeForCausalLM` that (a) registers, (b) loads NMC weights with correct fused layouts,
(c) passes your unit tests, (d) compiles a JAX forward (`load_format=dummy`) without error. Hand off to
the Performance Engineer for v7x execution.
