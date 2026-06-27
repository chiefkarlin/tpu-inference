# PORTING OVERVIEW — North Mini Code (Cohere2 MoE) → tpu-inference

## Target Model Profile
- **Model:** CohereLabs/North-Mini-Code-1.0 — Cohere2 MoE, 30.48B params (BF16), ~3B active/token.
- **Use case:** agentic code generation / SWE; 256K context, 64K max generation.
- **Architecture (`cohere2_moe` / `Cohere2MoeForCausalLM`):**
  - 49 transformer layers; `layer_types=[full,sliding,sliding,sliding]` ×12 + final `full`.
  - Layer 0 **dense** (intermediate=3072); layers 1–48 **MoE** (128 experts, 8 active, intermediate=768, **no shared experts**).
  - Attention: GQA 32 q-heads / 4 kv-heads, head_dim=128, no bias, no QK-norm.
  - **Sliding window** 4096 on sliding layers; full attention otherwise.
  - **Interleaved (NeoX) RoPE**, theta=50000, no scaling. Applied **only** on sliding layers + dense layer 0 (force_rope); **not** on full-attention MoE layers.
  - **MoE routing:** sigmoid selection, `norm_topk_prob=false` (raw sigmoid weights, no renormalization).
  - Gated SiLU experts: `down(silu(gate)*up) * weight`.
  - **Parallel transformer block:** `x + attn(norm(x)) + mlp(norm(x))`, single shared T5-style RMSNorm (eps=1e-6, no centering), no post-attention norm.
  - Embeddings 262144×2048; **lm_head tied** to embed_tokens; `logit_scale=1.0`.

## vLLM / tpu-inference Gap Analysis
- **Why custom porting is needed:** No Cohere2/North model exists in `tpu-inference`; the architecture combines a **parallel block** (no existing JAX precedent — all current models use sequential blocks), a **hybrid full/sliding schedule** with **conditional RoPE**, and **per-expert sharded weights** that must be fused into 3-D grouped-GEMM layout.
- **What is already supported (no new kernels):** sigmoid MoE routing + non-renorm top-k (fused_moe v1 + deepseek_v3), sliding-window attention (RPA v3 kernel), interleaved RoPE (`apply_rope` ordering="interleaved"), conditional per-layer RoPE (`use_attention_rope`), GQA, RMSNorm, gated SiLU, tied embeddings.
- **New work:** (1) `cohere2_moe.py` model file w/ parallel block + hybrid schedule; (2) weight remapper (128 per-expert → fused 3-D; dense layer 0; tied head); (3) registration in `model_loader.py`; (4) **verify** sliding-window reaches the RPA v3 kernel in decode; (5) v7x tuning of fused_moe + RPA block sizes.

## Proposed Architecture (JAX/Pallas)
- JAX-native `flax_nnx` impl (NMC not in `_VLLM_PREFERRED_ARCHITECTURES`, so `auto`→`flax_nnx`).
- Reuse `kernels/fused_moe/v1` + `kernels/megablox` grouped GEMM for MoE; `kernels/ragged_paged_attention/v3` for attention; `layers/jax/rope_interface.apply_rope(ordering="interleaved")` for RoPE.
- **TPU memory considerations (v7x-4, single host):** 128 experts × (2*768 + 768) × 2048 ≈ stacked expert weights dominate HBM (~expert params). The fused 3-D layout enables the grouped GEMM; SRAM block sizes must be tuned for the small 768 intermediate. Sliding window 4096 bounds KV working set per sliding layer; full layers retain full KV cache.
- **No new Pallas kernels required from scratch**; risk concentrated in (a) sliding-window decode wiring correctness and (b) v7x block-size tuning for the small-expert MoE GEMM.

## References
- Closest JAX references: `models/jax/deepseek_v3.py` (sigmoid routing/norm_topk_prob), `models/jax/gpt_oss.py` (per-layer full/sliding schedule), `models/jax/gemma4.py` & `llama4.py` (sliding window + interleaved RoPE).
- Detailed backlog: `experiments/north-mini-code/porting_backlog.md`.
- HF reference: `modeling_cohere2_moe.py` (transformers ≥5.8.0).
