# NMC Performance Analysis Findings (P-1 + P-2)

Code-verified analysis of the NMC MoE and attention kernel call paths,
with corrections to the original task-brief assumptions.

---

## P-1: MoE Tuning — Default Path is GMM_TP (NOT fused_moe/v1 or megablox)

### Original brief assumption (INCORRECT)
> "Tune fused_moe/v1 + megablox tuned_block_sizes.py for NMC MoE (E=128, I=768, D=2048, topk=8)"

### Actual call path (verified)
```
select_moe_backend(use_ep)              # layers/jax/moe/utils.py:259
  → GMM_TP (default; all env flags False, use_ep=False for single-host TP)
    → moe_apply()                       # layers/common/moe.py:94
      → fused_moe_func()                # layers/common/fused_moe_gmm.py:572
        → tensor_parallel_gmm()
          → moe_gmm_local()             # L182
            → gmm_wrapper()             # L123 (hardcodes tile_info=default)
              → gmm_v2()                # kernels/megablox/gmm_v2.py:1212
                → calculate_tiling()    # L884 (pure VMEM-fit heuristic, NO tuned table)
```

### Key findings
1. **The three tuned_block_sizes.py files are ALL OFF the default NMC path:**
   - `fused_moe/v1/tuned_block_sizes.py` → only FUSED_MOE backend (needs USE_MOE_EP_KERNEL=1 + EP).
   - `megablox/tuned_block_sizes.py` (gmm.py) → only MEGABLX_GMM backend (needs USE_UNFUSED_MEGABLOCKS=1).
     Also bypassed: `gmm_fn` passes explicit tiling from heuristic, not the table.
   - `gmm_v2` (actual NMC path) has NO tuned table — only `calculate_tiling` heuristic.

2. **Editing any tuned_block_sizes.py has ZERO effect on NMC's default MoE path.**

3. **`calculate_tiling` (gmm_v2.py:884-994)** is a pure VMEM-fitting heuristic:
   - tile_m=128 (bf16/bf16), tile_k=align_to(size_k, num_lanes), tile_n=align_to(size_n, num_lanes).
   - Shrinks tile_n then tile_k to fit vmem_limit=0.9×vmem_capacity.
   - For NMC bf16 unquantized: rhs_scale=None → has_scale=False, acc_dtype=float32.

4. **Real MoE tuning lever = `tile_info` param of gmm_v2**, which `gmm_wrapper`
   (fused_moe_gmm.py:123) hardcodes to `calculate_tiling` default.

### Orchestrator decision
Keep GMM_TP + calculate_tiling heuristic for baseline (correctness first).
Document tile_info tuning plan as P-1 follow-up (see PROFILING_PLAN.md §7).
No MoE code edits until baseline profiling proves MoE is a bottleneck.

---

## P-2: Attention Tuning — RPA v3 Clamps to sliding_window (NO over-fetch)

### Original brief assumption (INCORRECT)
> "Default heuristic ignores sliding_window → 2x wasted HBM BW on sliding layers"

### Actual behavior (verified in kernels/ragged_paged_attention/v3/kernel.py)
The regular RPA v3 kernel **clamps KV reads to sliding_window internally:**

- **Decode (L383-399):** `cur_seq_start_bkv_idx = max(kv_q_gap - sliding_window, 0) // bkv_sz`
  → KV fetch START clamped to sliding_window. Only sw tokens read.

- **Prefill/Mixed (L937-948):** `start_bkv_idx = max(processed_q_len - sw, 0) // bkv_sz`
  + `effective_kv_len = min(kv_len, processed_q_len + actual_bq_sz)`
  → effective KV range = [processed_q_len - sw, processed_q_len + bq_sz] = sliding window.

- **bkv loop is DYNAMIC:** `@pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)` (L956)
  → NOT static unroll, so large bkv_p does NOT bloat compiled code.

- **Per-iteration DMA clamped:** `effective_bkv_sz = min(effective_kv_len - bkv_idx*bkv_sz, bkv_sz)`
  (L1005) → HBM read ≈ sw tokens, NOT bkv_sz. **No over-fetch.**

### Default block_sizes (get_default_block_sizes L1496, case 7 / v7x)
- Decode: `bq_sz=1, bkv_sz=min(8192, max_kv), bq_csz=1, bkv_csz=min(8192, max_kv)`
- With page_size=16: bkv_p = bkv_sz/16 = 512, bkv_csz = 8192 tokens.

### VMEM pressure (the real concern)
- `bkv_double_buf = VMEM((2, bkv_sz, bkv_stride, *kv_cache.shape[3:]))` — scales with bkv_sz.
- With default bkv_sz=8192: ~8 MB/buffer × 2 (double-buffered) = ~16 MB for KV alone.
- Plus Q buffer, accumulator, l/m scratch → borderline on v7x VMEM (~64 MB/core).

### The regular tuned_block_sizes.py (4463 lines) is DEAD CODE
- `get_tuned_block_sizes` is NEVER called by kernel.py (only hd64 self-tunes via
  `get_tuned_block_sizes_hd64`).
- Populating the regular table has NO effect.

### Real P-2 lever = pass explicit block_sizes from NMC custom attention module
- Recommended decode: `d_block_sizes = (1, 4096, 1, 2048)`
  - bkv_sz=4096 = one full sliding window (50% smaller DMA buffer vs default 8192)
  - bkv_csz=2048 = 2 compute passes per window (good DMA/compute overlap)
  - All multiples of page_size=16 (validation constraint satisfied)
- Leave p_block_sizes/m_block_sizes=None (defaults) for now.

### page_size
- `PallasAttentionBackend.get_page_size` (flash_attn.py:86): if max_model_len > 8192 → page_size=16.
- NMC max_model_len=256K → **page_size=16** (may bump via get_min_page_size based on max_num_seqs).

### block_sizes API (for Kernel Eng)
- Kwargs: `d_block_sizes`, `p_block_sizes`, `m_block_sizes` (keyword-only on ragged_paged_attention)
- Tuple: `(bq_sz, bkv_sz, bq_csz, bkv_csz)`
- Validation: `bkv_sz % page_size == 0`, `bkv_csz % page_size == 0`, `bkv_sz % bkv_csz == 0`

### Coordination
Kernel Eng owns the custom attention module (cohere2_attention.py on feature/north-mini-code-kernels).
They will parameterize block_sizes so we can tune post-baseline without code changes.
K-1 (attention module with sliding_window forwarding) is DONE; K-3 (cohere2_moe.py) in progress.

---

## Summary of corrections to original backlog

| Item | Original claim | Verified reality |
|------|---------------|------------------|
| MoE path | fused_moe/v1 + megablox tuned tables | GMM_TP → gmm_v2 (no tuned table, heuristic only) |
| MoE tuning lever | Edit tuned_block_sizes.py | Wire tile_info through gmm_wrapper→gmm_v2 |
| Attention over-fetch | 2x HBM waste (ignores sw) | Kernel clamps to sw internally (no waste) |
| Attention tuned table | Populate regular tuned_block_sizes.py | Dead code; pass explicit block_sizes instead |
| Attention tuning lever | Table entries | Explicit d_block_sizes in custom attention module |
