#!/usr/bin/env python3
"""Sweep RPA v3 m_block_sizes / p_block_sizes for NMC prefill tuning.

This script calls the tpu_inference v3 ``ragged_paged_attention`` kernel
directly with NMC-shaped inputs and benchmarks different block size
configs at multiple prefill lengths. It must be run on a TPU node (v7x).

Usage:
    python scripts/sweep_rpa_block_sizes.py [--page-size 128] \\
        [--q-heads 8] [--kv-heads 1] [--head-dim 128] \\
        [--max-tokens 4096] [--num-warmup 5] [--num-iters 30]

NMC production shapes (TP=4, per-device):
    q_heads=8 (32/4), kv_heads=1 (4/4), head_dim=128, bf16, page_size=128

The v7x heuristic defaults for MIXED case (max_num_tokens=4096):
    bq_sz=256, bkv_sz=2048, bq_csz=128, bkv_csz=512

Block size constraints:
    bq_sz % bq_csz == 0
    bkv_sz % bkv_csz == 0
    bkv_sz % page_size == 0
    bkv_csz % page_size == 0
"""

import argparse
import functools
import math
import sys
import time
import traceback
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    ragged_paged_attention,
    get_default_block_sizes,
    RpaCase,
)
from jax.experimental.pallas import tpu as pltpu


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class SweepConfig(NamedTuple):
    name: str
    m_block_sizes: tuple | None  # None = heuristic default
    chunk_prefill_size: int | None  # None = MIXED only; int = enable PREFILL
    p_block_sizes: tuple | None  # Only used when chunk_prefill_size is not None


def get_default_m_block_sizes(q_heads, kv_heads, head_dim, page_size,
                              max_tokens, pages_per_seq):
    """Get the v7x heuristic default for MIXED case."""
    d = get_default_block_sizes(
        jnp.bfloat16, jnp.bfloat16,
        q_heads, kv_heads, head_dim, page_size,
        max_tokens, 1, pages_per_seq,
        case=RpaCase.MIXED,
    )
    return (d["bq_sz"], d["bkv_sz"], d["bq_csz"], d["bkv_csz"])


def build_sweep_configs(default_bs, page_size):
    """Build the list of configs to sweep.

    Approach A: vary m_block_sizes (MIXED case, chunk_prefill_size=None).
    Approach B: enable dedicated PREFILL (chunk_prefill_size + p_block_sizes).
    """
    dq, dkv, dqc, dkvc = default_bs
    configs = []

    # --- Approach A: m_block_sizes sweep (MIXED case) ---
    # Default (heuristic)
    configs.append(SweepConfig("default", default_bs, None, None))

    # Vary bq_sz (query fetch block)
    for bq in [128, 512, 1024]:
        if bq % dqc == 0:
            configs.append(SweepConfig(
                f"m_bq{bq}", (bq, dkv, dqc, dkvc), None, None))

    # Vary bkv_sz (kv fetch block)
    for bkv in [1024, 4096]:
        if bkv % page_size == 0 and bkv % dkvc == 0:
            configs.append(SweepConfig(
                f"m_bkv{bkv}", (dq, bkv, dqc, dkvc), None, None))

    # Vary bq_csz (query compute chunk)
    for bqc in [64, 256]:
        if dq % bqc == 0:
            configs.append(SweepConfig(
                f"m_bqc{bqc}", (dq, dkv, bqc, dkvc), None, None))

    # Vary bkv_csz (kv compute chunk)
    for bkvc in [256, 1024]:
        if dkv % bkvc == 0 and bkvc % page_size == 0:
            configs.append(SweepConfig(
                f"m_bkvc{bkvc}", (dq, dkv, dqc, bkvc), None, None))

    # Combined configs
    combined = [
        (512, 4096, 256, 1024),   # large everything
        (512, 2048, 256, 512),    # larger q
        (256, 4096, 128, 1024),   # larger kv
        (1024, 2048, 512, 512),   # very large q
        (256, 1024, 128, 256),    # smaller kv
        (128, 1024, 64, 256),     # small everything
    ]
    for bs in combined:
        bq, bkv, bqc, bkvc = bs
        if (bq % bqc == 0 and bkv % bkvc == 0
                and bkv % page_size == 0 and bkvc % page_size == 0):
            configs.append(SweepConfig(f"m_combo_{bq}_{bkv}_{bqc}_{bkvc}",
                                       bs, None, None))

    # --- Approach B: dedicated PREFILL (chunk_prefill_size set) ---
    # chunk_prefill_size is set to prefill_len per benchmark (see
    # benchmark_config) to avoid the KV corruption bug. Here we just mark
    # the config as "use prefill kernel" and vary p_block_sizes.
    # Use heuristic defaults for p_block_sizes.
    configs.append(SweepConfig("p_default", None, "prefill", None))
    # Try a few p_block_sizes
    for bs in [(256, 2048, 128, 512), (512, 2048, 256, 512),
                (256, 4096, 128, 1024), (512, 4096, 256, 1024),
                (1024, 2048, 512, 512), (128, 1024, 64, 256)]:
        bq, bkv, bqc, bkvc = bs
        if (bq % bqc == 0 and bkv % bkvc == 0
                and bkv % page_size == 0 and bkvc % page_size == 0):
            configs.append(SweepConfig(
                f"p_{bq}_{bkv}_{bqc}_{bkvc}", None, "prefill", bs))

    return configs


# ---------------------------------------------------------------------------
# Input creation
# ---------------------------------------------------------------------------

def create_prefill_inputs(q_heads, kv_heads, head_dim, page_size,
                          max_tokens, prefill_len, *, use_prefill_kernel=False,
                          dtype=jnp.bfloat16):
    """Create RPA v3 inputs for a single-sequence prefill.

    Returns (queries, keys, values, kv_cache, kv_lens, page_indices,
             cu_q_lens, distribution) matching the public API.

    Args:
        use_prefill_kernel: If True, route to PREFILL kernel (distribution
            = (0,1,1)). If False, route to MIXED kernel (distribution =
            (0,0,1)).
    """
    key = jax.random.key(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Queries: [max_num_tokens, num_q_heads, head_dim]
    q = jax.random.normal(k1, (max_tokens, q_heads, head_dim), dtype=dtype)

    # Keys/Values: [max_num_tokens, num_kv_heads, head_dim]
    k = jax.random.normal(k2, (max_tokens, kv_heads, head_dim), dtype=dtype)
    v = jax.random.normal(k3, (max_tokens, kv_heads, head_dim), dtype=dtype)

    # KV cache: [total_num_pages, page_size, num_kv_heads_x2 // kv_packing,
    #            kv_packing, head_dim]
    kv_packing = 2 if dtype == jnp.bfloat16 else 1
    num_kv_heads_x2 = kv_heads * 2
    num_kv_heads_x2_per_packing = num_kv_heads_x2 // kv_packing
    pages_per_seq = math.ceil(max_tokens / page_size)
    total_num_pages = pages_per_seq  # single sequence
    kv_cache = jax.random.normal(
        k4,
        (total_num_pages, page_size, num_kv_heads_x2_per_packing,
         kv_packing, head_dim),
        dtype=dtype,
    )

    # Metadata
    kv_lens = jnp.array([prefill_len], dtype=jnp.int32)
    page_indices = jnp.arange(total_num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, prefill_len], dtype=jnp.int32)
    # distribution = (i, j, k): seqs[0:i] decode, [i:j] prefill, [j:k] mixed
    # For prefill via MIXED: (0, 0, 1) — all seqs are mixed
    # For prefill via PREFILL: (0, 1, 1) — 1 seq is prefill, 0 mixed
    if use_prefill_kernel:
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
    else:
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    return q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_config(config, q_heads, kv_heads, head_dim, page_size,
                     max_tokens, prefill_lengths, num_warmup, num_iters):
    """Benchmark a single block size config at multiple prefill lengths.

    Returns dict: {prefill_len: median_ms} or {"error": msg}.
    """
    sm_scale = 1.0 / math.sqrt(head_dim)
    pages_per_seq = math.ceil(max_tokens / page_size)
    results = {}

    for plen in prefill_lengths:
        if plen > max_tokens:
            continue

        # For Approach B (PREFILL kernel): route to PREFILL via distribution,
        # and set chunk_prefill_size = plen to avoid the KV corruption bug
        # (static_q_len must match the actual q_len).
        use_prefill_kernel = config.chunk_prefill_size is not None
        inputs = create_prefill_inputs(
            q_heads, kv_heads, head_dim, page_size,
            max_tokens, plen,
            use_prefill_kernel=use_prefill_kernel,
        )
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, dist = inputs

        # Build kwargs
        kwargs = dict(
            sm_scale=sm_scale,
            sliding_window=4096,  # NMC sliding layers
            use_causal_mask=True,
            update_kv_cache=True,
        )
        if config.m_block_sizes is not None:
            kwargs["m_block_sizes"] = config.m_block_sizes
        if use_prefill_kernel:
            # Set chunk_prefill_size = plen so static_q_len matches actual
            # q_len (avoids reading beyond valid q tokens).
            kwargs["chunk_prefill_size"] = plen
            if config.p_block_sizes is not None:
                kwargs["p_block_sizes"] = config.p_block_sizes

        # JIT-compile
        def fn(q, k, v, kv_cache, kv_lens, pi, cq, d):
            return ragged_paged_attention(
                q, k, v, kv_cache, kv_lens, pi, cq, d, **kwargs)

        try:
            jitted = jax.jit(fn)
            # Warmup (includes compile)
            for _ in range(num_warmup):
                out = jitted(q, k, v, kv_cache, kv_lens, page_indices,
                             cu_q_lens, dist)
                out[0].block_until_ready()

            # Benchmark
            times = []
            for _ in range(num_iters):
                t0 = time.perf_counter()
                out = jitted(q, k, v, kv_cache, kv_lens, page_indices,
                             cu_q_lens, dist)
                out[0].block_until_ready()
                times.append(time.perf_counter() - t0)

            times_ms = np.array(times) * 1000
            results[plen] = {
                "median_ms": float(np.median(times_ms)),
                "mean_ms": float(np.mean(times_ms)),
                "std_ms": float(np.std(times_ms)),
                "min_ms": float(np.min(times_ms)),
            }
        except Exception as e:
            results[plen] = {"error": str(e)[:200]}
            # Don't re-raise; continue with other lengths

        # Clear cache to avoid cross-config interference
        jax.clear_caches()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep RPA v3 block sizes for NMC prefill tuning")
    parser.add_argument("--page-size", type=int, default=128,
                        help="Page size (default: 128, cluster may bump to 256)")
    parser.add_argument("--q-heads", type=int, default=8,
                        help="Q heads per device (default: 8 = 32/4 TP)")
    parser.add_argument("--kv-heads", type=int, default=1,
                        help="KV heads per device (default: 1 = 4/4 TP)")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max num batched tokens (compile-time shape)")
    parser.add_argument("--prefill-lengths", type=str, default="128,512,1024,2048,4096",
                        help="Comma-separated prefill lengths to benchmark")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path")
    args = parser.parse_args()

    prefill_lengths = [int(x) for x in args.prefill_lengths.split(",")]
    pages_per_seq = math.ceil(args.max_tokens / args.page_size)

    print(f"=== RPA v3 Block Size Sweep for NMC Prefill ===")
    print(f"TPU: {jax.devices()}")
    print(f"Shapes: q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
          f"head_dim={args.head_dim}, page_size={args.page_size}, "
          f"max_tokens={args.max_tokens}")
    print(f"Prefill lengths: {prefill_lengths}")
    print(f"Warmup: {args.num_warmup}, Iters: {args.num_iters}")
    print()

    # Get heuristic defaults
    default_bs = get_default_m_block_sizes(
        args.q_heads, args.kv_heads, args.head_dim, args.page_size,
        args.max_tokens, pages_per_seq)
    print(f"Heuristic MIXED defaults: {default_bs}")
    print()

    # Build sweep configs
    configs = build_sweep_configs(default_bs, args.page_size)
    print(f"Sweeping {len(configs)} configs...")
    print()

    # Run sweep
    all_results = []
    for i, config in enumerate(configs):
        label = config.name
        if config.m_block_sizes is not None:
            label += f" m={config.m_block_sizes}"
        if config.chunk_prefill_size is not None:
            label += f" cps={config.chunk_prefill_size}"
            if config.p_block_sizes is not None:
                label += f" p={config.p_block_sizes}"
        print(f"[{i+1}/{len(configs)}] {label}")

        results = benchmark_config(
            config, args.q_heads, args.kv_heads, args.head_dim,
            args.page_size, args.max_tokens, prefill_lengths,
            args.num_warmup, args.num_iters)

        row = {"config": config.name, "m_block_sizes": str(config.m_block_sizes),
               "chunk_prefill_size": str(config.chunk_prefill_size),
               "p_block_sizes": str(config.p_block_sizes)}
        for plen in prefill_lengths:
            if plen in results and "error" not in results[plen]:
                row[f"ttft_{plen}_ms"] = f"{results[plen]['median_ms']:.3f}"
                print(f"    L={plen:5d}: {results[plen]['median_ms']:.3f}ms")
            elif plen in results:
                row[f"ttft_{plen}_ms"] = "ERROR"
                print(f"    L={plen:5d}: ERROR - {results[plen]['error'][:80]}")
            else:
                row[f"ttft_{plen}_ms"] = "-"
        all_results.append(row)
        print()

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Config':<35}", end="")
    for plen in prefill_lengths:
        print(f" L={plen:>5d}", end="")
    print()
    print("-" * (35 + 12 * len(prefill_lengths)))

    for row in all_results:
        print(f"{row['config']:<35}", end="")
        for plen in prefill_lengths:
            val = row.get(f"ttft_{plen}_ms", "-")
            print(f" {val:>10}", end="")
        print()

    # Find best config per length
    print("\n=== BEST CONFIG PER PREFILL LENGTH ===")
    for plen in prefill_lengths:
        best_config = None
        best_time = float("inf")
        for row in all_results:
            val = row.get(f"ttft_{plen}_ms", "-")
            if val not in ("-", "ERROR", "None"):
                t = float(val)
                if t < best_time:
                    best_time = t
                    best_config = row["config"]
        if best_config:
            print(f"  L={plen:>5d}: {best_config} = {best_time:.3f}ms")

    # CSV output
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCSV saved to {args.output}")


if __name__ == "__main__":
    main()
