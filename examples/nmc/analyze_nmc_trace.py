#!/usr/bin/env python3
"""NMC trace analysis script for P-4 profiling.

Runs inside the Docker container (which has MaxKernel tooling + tensorflow +
xprof). Analyzes an xplane.pb trace from a v7x-4 NMC inference run and produces
a markdown report with:
  1. Overview metrics (step time, duty cycle)
  2. Top ops by total duration
  3. NMC-specific kernel timings (attention, MoE GEMM)
  4. Compute vs memory ratio (SyncWait fraction)
  5. Comparison vs HBM-bound theoretical minimums

Usage:
  python3 analyze_nmc_trace.py <xplane.pb path> [--output report.md]

The xplane.pb is captured by setting JAX_TRACE_START_GPU / JAX_TRACE_STOP_GPU
env vars in the pod, or via jax.profiler.trace() in a benchmark script.
"""

import argparse
import json
import os
import sys

# MaxKernel offline tooling (installed in Docker image via accelerator-agents).
# These imports require tensorflow (for xplane_pb2 proto) and xprof.
sys.path.insert(0, "/workspace/accelerator-agents/MaxKernel")

try:
    from hitl_agent.subagents.profiling import offline_tools
    from hitl_agent.tools.analyze_profile import analyze_trace
except ImportError:
    # Fallback: try the auto_agent copy (byte-identical).
    try:
        from auto_agent.subagents.profiling import offline_tools
        from auto_agent.tools.analyze_profile import analyze_trace
    except ImportError:
        print("ERROR: MaxKernel tooling not found. This script must run inside"
              " the Docker image with /workspace/accelerator-agents mounted.",
              file=sys.stderr)
        sys.exit(1)


# --- NMC-specific query templates ---

# Top 30 ops by total duration (picoseconds → ms).
TOP_OPS_SQL = """
SELECT name,
       COUNT(*) AS num_calls,
       SUM(duration_ps) AS total_ps,
       AVG(duration_ps) AS avg_ps,
       MAX(duration_ps) AS max_ps
FROM events
GROUP BY name
ORDER BY total_ps DESC
LIMIT 30;
"""

# Attention ops (ragged_paged_attention). The RPA v3 kernel scope name is:
#   RPA{D|P|M}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}[-sw_{sliding_window}]
# so we filter for "RPA" or "ragged_paged_attention".
ATTENTION_SQL = """
SELECT name,
       COUNT(*) AS num_calls,
       SUM(duration_ps) AS total_ps,
       AVG(duration_ps) AS avg_ps
FROM events
WHERE name LIKE '%ragged_paged%'
   OR name LIKE '%RPA%'
   OR name LIKE '%attention%'
GROUP BY name
ORDER BY total_ps DESC
LIMIT 20;
"""

# MoE GEMM ops (gmm_v2 / megablox / fused_moe).
MOE_SQL = """
SELECT name,
       COUNT(*) AS num_calls,
       SUM(duration_ps) AS total_ps,
       AVG(duration_ps) AS avg_ps
FROM events
WHERE name LIKE '%gmm%'
   OR name LIKE '%megablox%'
   OR name LIKE '%fused_moe%'
   OR name LIKE '%moe%'
GROUP BY name
ORDER BY total_ps DESC
LIMIT 20;
"""

# SyncWait / DMA stall events (memory-bound indicator).
SYNCWAIT_SQL = """
SELECT name,
       COUNT(*) AS num_calls,
       SUM(duration_ps) AS total_ps,
       AVG(duration_ps) AS avg_ps
FROM events
WHERE name LIKE '%SyncWait%'
   OR name LIKE '%DMA%'
   OR name LIKE '%Copy%'
GROUP BY name
ORDER BY total_ps DESC
LIMIT 20;
"""

# All jit_computation / pallas_call events (the top-level kernel invocations).
JIT_SQL = """
SELECT name,
       COUNT(*) AS num_calls,
       SUM(duration_ps) AS total_ps,
       AVG(duration_ps) AS avg_ps
FROM events
WHERE name LIKE '%jit_computation%'
   OR name LIKE '%pallas%'
   OR name LIKE '%xla%'
GROUP BY name
ORDER BY total_ps DESC
LIMIT 20;
"""


def ps_to_ms(ps: float) -> float:
    """Picoseconds to milliseconds."""
    return ps / 1e9


def ps_to_us(ps: float) -> float:
    """Picoseconds to microseconds."""
    return ps / 1e6


def run_query(xplane_path: str, sql: str, title: str) -> str:
    """Run a SQL query against the xplane and return a markdown section."""
    try:
        result = offline_tools.load_xplane_and_query(xplane_path, sql)
        return f"\n### {title}\n\n{result}\n"
    except Exception as e:
        return f"\n### {title}\n\nERROR: {e}\n"


def get_compute_memory_ratio(xplane_path: str) -> str:
    """Compute the SyncWait ratio (memory-bound fraction)."""
    try:
        ratio = analyze_trace(xplane_path)
        if ratio is None:
            return ("### Compute vs Memory Ratio\n\n"
                    "analyze_trace returned None (no jit_computation events "
                    "found — trace may be empty or from wrong device).\n")
        pct_wait = ratio * 100
        pct_compute = (1 - ratio) * 100
        verdict = ("MEMORY-BOUND (SyncWait dominant)"
                   if ratio > 0.5
                   else "COMPUTE-BOUND" if ratio < 0.3
                   else "BALANCED")
        return (f"### Compute vs Memory Ratio\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| SyncWait fraction | {pct_wait:.2f}% |\n"
                f"| Compute fraction | {pct_compute:.2f}% |\n"
                f"| Verdict | **{verdict}** |\n\n"
                f"SyncWait fraction > 50% indicates the kernel is stalled on "
                f"DMA/memory transfers (HBM-bound). < 30% indicates compute-"
                f"bound (MXU utilization matters more).\n")
    except Exception as e:
        return f"### Compute vs Memory Ratio\n\nERROR: {e}\n"


def get_overview(xplane_path: str) -> str:
    """Get high-level trace metrics."""
    try:
        result = offline_tools.get_overview_page_metrics(xplane_path)
        metrics = json.loads(result)
        lines = ["### Overview Metrics\n\n| Metric | Value |\n|--------|-------|\n"]
        for k, v in metrics.items():
            lines.append(f"| {k} | {v} |\n")
        return "\n".join(lines) + "\n"
    except Exception as e:
        return f"### Overview Metrics\n\nERROR: {e}\n"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NMC v7x-4 inference trace.")
    parser.add_argument("xplane_path",
                        help="Path to the xplane.pb trace file.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output markdown file (default: stdout).")
    args = parser.parse_args()

    if not os.path.exists(args.xplane_path):
        print(f"ERROR: {args.xplane_path} not found", file=sys.stderr)
        sys.exit(1)

    xplane_path = args.xplane_path
    print(f"Analyzing trace: {xplane_path}", file=sys.stderr)

    report = []
    report.append("# NMC v7x-4 Trace Analysis Report\n")
    report.append(f"**Trace:** `{xplane_path}`\n")

    # 1. Overview
    report.append(get_overview(xplane_path))

    # 2. Compute vs memory ratio
    report.append(get_compute_memory_ratio(xplane_path))

    # 3. Top ops by duration
    report.append(run_query(xplane_path, TOP_OPS_SQL,
                            "Top 30 Ops by Total Duration"))

    # 4. JIT/pallas computations
    report.append(run_query(xplane_path, JIT_SQL,
                            "JIT/Pallas Computations"))

    # 5. Attention kernels
    report.append(run_query(xplane_path, ATTENTION_SQL,
                            "Attention (RPA) Kernels"))

    # 6. MoE GEMM kernels
    report.append(run_query(xplane_path, MOE_SQL,
                            "MoE GEMM Kernels"))

    # 7. SyncWait / DMA stalls
    report.append(run_query(xplane_path, SYNCWAIT_SQL,
                            "SyncWait / DMA Stall Events"))

    # 8. HBM-bound reference (decode)
    report.append("""
### HBM-Bound Reference (Decode, per token)

Theoretical minimum decode latency (full model, TP=4):
  T_min = total_HBM_bytes_per_chip / HBM_BW_per_chip

NMC decode reads (per chip, TP=4, bf16):
  - Dense layer 0: attn + MLP weights
  - 48 MoE layers: 8 active experts × (gate+up+down) + attn + router
  - lm_head (tied embed)

If measured decode step time >> T_min, the gap is:
  - SyncWait stalls (DMA latency, poor overlap)
  - Underutilized HBM BW (small block sizes, poor tiling)
  - Compute overhead (routing, activation, etc.)

See docs/north-mini-code/PROFILING_PLAN.md §3 for the detailed byte breakdown.
""")

    output = "\n".join(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
