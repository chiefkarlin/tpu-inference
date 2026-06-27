#!/usr/bin/env python3
"""NMC trace analysis script for P-4 profiling.

Self-contained: requires only ``tensorflow`` (for the xplane_pb2 proto) and
``pandas``.  No MaxKernel or xprof dependency.

Analyzes an xplane.pb trace from a v7x-4 NMC inference run and produces a
markdown report with:
  1. Overview metrics (step time, duty cycle)
  2. Top ops by total duration
  3. NMC-specific kernel timings (attention, MoE GEMM)
  4. Compute vs memory ratio (SyncWait fraction — computed via SQL)
  5. Comparison vs HBM-bound theoretical minimums

Usage:
  pip install tensorflow pandas
  python3 analyze_nmc_trace.py <xplane.pb path> [--output report.md]

The xplane.pb is captured via vLLM's profiling API (POST /start_profile,
POST /stop_profile) or the offline ``examples/tpu_profiling.py`` script.
See docs/north-mini-code/EXECUTION_RUNBOOK.md §6 for the full capture flow.
"""

import argparse
import gzip
import json
import os
import sqlite3
import sys

try:
    import pandas as pd
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
except ImportError:
    print(
        "ERROR: tensorflow and pandas are required.\n"
        "  pip install tensorflow pandas",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# xplane.pb → in-memory SQLite (inlined from MaxKernel offline_tools.py)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE planes (id INTEGER, name TEXT);
CREATE TABLE lines (
    id INTEGER, plane_id INTEGER, display_id INTEGER,
    name TEXT, timestamp_ns INTEGER
);
CREATE TABLE events (
    plane_id INTEGER, line_id INTEGER,
    name TEXT, offset_ps INTEGER, duration_ps INTEGER,
    start_ps INTEGER, end_ps INTEGER
);
"""


def _load_xspace(xplane_path: str) -> xplane_pb2.XSpace:
    """Load an xplane.pb (or .gz) file into an XSpace proto."""
    open_func = gzip.open if xplane_path.endswith(".gz") else open
    with open_func(xplane_path, "rb") as f:
        xspace = xplane_pb2.XSpace()
        xspace.ParseFromString(f.read())
    return xspace


def build_sqlite_db(xplane_path: str) -> sqlite3.Connection:
    """Parse xplane.pb and populate an in-memory SQLite database."""
    xspace = _load_xspace(xplane_path)
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.executescript(SCHEMA_SQL)

    for plane in xspace.planes:
        c.execute("INSERT INTO planes VALUES (?, ?)", (plane.id, plane.name))
        meta = plane.event_metadata  # id → EventMetadata(name=...)

        for line in plane.lines:
            c.execute(
                "INSERT INTO lines VALUES (?, ?, ?, ?, ?)",
                (line.id, plane.id, line.display_id, line.name,
                 line.timestamp_ns),
            )
            for event in line.events:
                name = (meta[event.metadata_id].name
                        if event.metadata_id in meta
                        else str(event.metadata_id))
                start_ps = event.offset_ps
                c.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (plane.id, line.id, name,
                     event.offset_ps, event.duration_ps,
                     start_ps, start_ps + event.duration_ps),
                )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# SQL query templates
# ---------------------------------------------------------------------------

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

# SyncWait ratio: fraction of time in SyncWait events within the last two
# jit_computation events on /device:TPU:0.  This replaces MaxKernel's
# xprof-dependent analyze_trace() with a pure-SQL computation.
SYNCWAIT_RATIO_SQL = """
WITH tpu_plane AS (
    SELECT id FROM planes WHERE name LIKE '%/device:TPU:0%' LIMIT 1
),
jit_events AS (
    SELECT e.start_ps, e.end_ps
    FROM events e
    JOIN tpu_plane tp ON e.plane_id = tp.id
    WHERE e.name LIKE '%jit_computation%'
    ORDER BY e.start_ps DESC
    LIMIT 2
),
window AS (
    SELECT
        (SELECT start_ps FROM jit_events ORDER BY start_ps DESC LIMIT 1 OFFSET 1)
            AS start_last,
        (SELECT end_ps   FROM jit_events ORDER BY start_ps DESC LIMIT 1)
            AS end_last
),
sync_in_window AS (
    SELECT COALESCE(SUM(e.duration_ps), 0) AS sync_wait_ps
    FROM events e
    JOIN tpu_plane tp ON e.plane_id = tp.id
    CROSS JOIN window w
    WHERE e.name LIKE '%SyncWait%'
      AND e.start_ps >= w.start_last
      AND e.end_ps   <= w.end_last
)
SELECT
    w.start_last,
    w.end_last,
    (w.end_last - w.start_last) AS total_ps,
    s.sync_wait_ps,
    CASE WHEN (w.end_last - w.start_last) > 0
         THEN CAST(s.sync_wait_ps AS FLOAT) / (w.end_last - w.start_last)
         ELSE 0
    END AS ratio
FROM window w, sync_in_window s;
"""


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def ps_to_ms(ps: float) -> float:
    return ps / 1e9


def run_query(conn: sqlite3.Connection, sql: str, title: str,
              convert_ps: bool = True) -> str:
    try:
        df = pd.read_sql_query(sql, conn)
        if convert_ps:
            for col in ("total_ps", "avg_ps", "max_ps"):
                if col in df.columns:
                    df[col] = df[col].apply(ps_to_ms).round(4)
            df = df.rename(columns={
                "total_ps": "total_ms", "avg_ps": "avg_ms",
                "max_ps": "max_ms",
            })
        table = df.to_markdown(index=False)
        return f"\n### {title}\n\n{table}\n"
    except Exception as e:
        return f"\n### {title}\n\nERROR: {e}\n"


def get_compute_memory_ratio(conn: sqlite3.Connection) -> str:
    try:
        c = conn.cursor()
        row = c.execute(SYNCWAIT_RATIO_SQL).fetchone()
        if row is None or row[3] is None:
            return ("### Compute vs Memory Ratio\n\n"
                    "No jit_computation events found on /device:TPU:0 — "
                    "trace may be empty or from the wrong device.\n")
        _start, _end, total_ps, sync_ps, ratio = row
        pct_wait = ratio * 100
        pct_compute = (1 - ratio) * 100
        verdict = ("MEMORY-BOUND (SyncWait dominant)"
                   if ratio > 0.5
                   else "COMPUTE-BOUND" if ratio < 0.3
                   else "BALANCED")
        return (
            "### Compute vs Memory Ratio\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Window total | {ps_to_ms(total_ps):.4f} ms |\n"
            f"| SyncWait total | {ps_to_ms(sync_ps):.4f} ms |\n"
            f"| SyncWait fraction | {pct_wait:.2f}% |\n"
            f"| Compute fraction | {pct_compute:.2f}% |\n"
            f"| Verdict | **{verdict}** |\n\n"
            "SyncWait fraction > 50% = stalled on DMA/memory transfers "
            "(HBM-bound). < 30% = compute-bound (MXU utilization matters "
            "more). Computed over the last two jit_computation events on "
            "/device:TPU:0.\n"
        )
    except Exception as e:
        return f"### Compute vs Memory Ratio\n\nERROR: {e}\n"


def get_overview(xplane_path: str) -> str:
    try:
        xspace = _load_xspace(xplane_path)
        device_planes, host_planes = [], []
        for p in xspace.planes:
            if any(k in p.name.lower()
                   for k in ("device", "tpu", "gpu")):
                device_planes.append(p)
            else:
                host_planes.append(p)

        min_start, max_end = float("inf"), 0
        for p in host_planes + device_planes:
            for line in p.lines:
                for e in line.events:
                    min_start = min(min_start, e.offset_ps)
                    max_end = max(max_end, e.offset_ps + e.duration_ps)
        total_ms = ps_to_ms(max_end - min_start) if max_end > min_start else 0

        duty = 0.0
        if device_planes and max_end > min_start:
            busy = sum(e.duration_ps for p in device_planes
                       for line in p.lines for e in line.events)
            potential = len(device_planes) * (max_end - min_start)
            duty = (busy / potential * 100) if potential > 0 else 0

        step_count, step_durs = 0, []
        for p in host_planes + device_planes:
            for line in p.lines:
                if "steps" in line.name.lower():
                    for e in line.events:
                        step_count += 1
                        step_durs.append(e.duration_ps)
        avg_step_ms = (ps_to_ms(sum(step_durs) / step_count)
                       if step_count > 0 else 0)

        lines = ["### Overview Metrics\n\n| Metric | Value |\n|---|---|\n"]
        for k, v in [("device_count", len(device_planes)),
                      ("host_count", len(host_planes)),
                      ("total_duration_ms", f"{total_ms:.2f}"),
                      ("device_duty_cycle_percent", f"{duty:.2f}"),
                      ("average_step_time_ms", f"{avg_step_ms:.4f}"),
                      ("step_count", step_count)]:
            lines.append(f"| {k} | {v} |\n")
        return "\n".join(lines) + "\n"
    except Exception as e:
        return f"### Overview Metrics\n\nERROR: {e}\n"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NMC v7x-4 inference trace (self-contained).")
    parser.add_argument("xplane_path",
                        help="Path to the xplane.pb trace file.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output markdown file (default: stdout).")
    args = parser.parse_args()

    if not os.path.exists(args.xplane_path):
        print(f"ERROR: {args.xplane_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing trace: {args.xplane_path}", file=sys.stderr)
    conn = build_sqlite_db(args.xplane_path)

    report = [
        "# NMC v7x-4 Trace Analysis Report\n",
        f"**Trace:** `{args.xplane_path}`\n",
        get_overview(args.xplane_path),
        get_compute_memory_ratio(conn),
        run_query(conn, TOP_OPS_SQL, "Top 30 Ops by Total Duration"),
        run_query(conn, JIT_SQL, "JIT/Pallas Computations"),
        run_query(conn, ATTENTION_SQL, "Attention (RPA) Kernels"),
        run_query(conn, MOE_SQL, "MoE GEMM Kernels"),
        run_query(conn, SYNCWAIT_SQL, "SyncWait / DMA Stall Events"),
        """
### HBM-Bound Reference (Decode, per token)

Theoretical minimum decode latency (full model, TP=4):
  T_min = total_HBM_bytes_per_chip / HBM_BW_per_chip

NMC decode reads (per chip, TP=4, bf16, ctx=8192):
  - Dense layer 0: attn + MLP weights (replicated)
  - 48 MoE layers: 8/128 active experts × (gate+up+down) + attn + router
  - lm_head (tied embed, vocab/TP sharded)
  - KV cache (sliding layers: sw=4096; full layers: full ctx)

Corrected per-chip decode HBM: ~1.87 GB
  T_min = 1.87 GB / 7.4 TB/s = 0.253 ms/token (~3,950 tok/s if HBM-bound)

See docs/north-mini-code/PROFILING_PLAN.md §3-§4 for the detailed breakdown.

If measured decode step time >> T_min, the gap is:
  - SyncWait stalls (DMA latency, poor overlap)
  - Underutilized HBM BW (small block sizes, poor tiling)
  - Compute overhead (routing, activation, all-reduce)
""",
    ]
    conn.close()

    output = "\n".join(report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
