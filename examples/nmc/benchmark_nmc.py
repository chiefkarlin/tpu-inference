#!/usr/bin/env python3
"""
NMC TPU Benchmark Script — produces metrics comparable to GPU implementations.

Runs vllm bench serve against the running NMC endpoint (localhost:8000) with
controlled synthetic workloads. Collects TTFT, TPOT, ITL, and throughput metrics.

Usage:
    python3 benchmark_nmc.py [--base-url http://localhost:8000] [--result-dir /tmp/nmc-bench]

Benchmark matrix:
    1. Prefill latency (TTFT): varying input lengths, output=1, sequential
    2. Decode latency (TPOT): fixed input, varying output lengths, sequential
    3. Throughput scaling: fixed input/output, varying concurrency (burst)
    4. Realistic mixed: sonnet-like 512/256, concurrency sweep
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_bench(base_url, model, dataset, input_len, output_len, num_prompts,
              max_concurrency=None, request_rate="inf", ignore_eos=True,
              result_dir="/tmp/nmc-bench", label=None, extra_args=None):
    """Run a single vllm bench serve invocation and return parsed results."""
    if label is None:
        label = f"in{input_len}_out{output_len}_n{num_prompts}_c{max_concurrency or num_prompts}"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "vllm",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", dataset,
        "--endpoint", "/v1/completions",
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--result-dir", result_dir,
        "--result-filename", f"{label}.json",
        "--metric-percentiles", "50,90,99",
    ]

    if dataset == "random":
        cmd += ["--random-input-len", str(input_len), "--random-output-len", str(output_len)]
    else:
        cmd += ["--input-len", str(input_len), "--output-len", str(output_len)]

    if max_concurrency is not None:
        cmd += ["--max-concurrency", str(max_concurrency)]

    if ignore_eos:
        cmd += ["--ignore-eos"]

    if extra_args:
        cmd += extra_args

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {label}")
    print(f"  input_len={input_len}, output_len={output_len}, num_prompts={num_prompts}, "
          f"concurrency={max_concurrency or 'unlimited'}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        print(f"STDERR: {result.stderr[-500:]}")
        return {"label": label, "status": "failed", "error": result.stderr[-500:],
                "elapsed_s": elapsed}

    # Parse the JSON result file
    result_file = Path(result_dir) / f"{label}.json"
    parsed = {"label": label, "status": "success", "elapsed_s": elapsed}
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        # Extract key metrics
        for key in ["ttft", "tpot", "itl", "request_latency",
                     "output_throughput", "request_throughput", "total_token_throughput"]:
            if key in data:
                parsed[key] = data[key]

    # Also extract from stdout (vllm bench prints a summary)
    for line in result.stdout.split("\n"):
        line = line.strip()
        if any(k in line.lower() for k in ["throughput", "latency", "ttft", "tpot", "tokens/s"]):
            print(f"  {line}")

    print(f"  Completed in {elapsed:.1f}s")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="NMC TPU Benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="CohereLabs/North-Mini-Code-1.0")
    parser.add_argument("--result-dir", default="/tmp/nmc-bench")
    parser.add_argument("--phase", default="all",
                        choices=["all", "prefill", "decode", "throughput", "mixed"])
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    all_results = []

    # ---- Phase 1: Prefill latency (TTFT) ----
    # Varying input length, output=1, sequential (concurrency=1)
    if args.phase in ("all", "prefill"):
        print("\n" + "="*70)
        print("PHASE 1: PREFILL LATENCY (TTFT)")
        print("="*70)
        for input_len in [128, 512, 1024, 2048, 4096]:
            r = run_bench(
                args.base_url, args.model, "random",
                input_len=input_len, output_len=1,
                num_prompts=10, max_concurrency=1,
                request_rate="inf", ignore_eos=True,
                result_dir=args.result_dir,
                label=f"prefill_in{input_len}",
            )
            all_results.append(r)

    # ---- Phase 2: Decode latency (TPOT) ----
    # Fixed short input, varying output length, sequential
    if args.phase in ("all", "decode"):
        print("\n" + "="*70)
        print("PHASE 2: DECODE LATENCY (TPOT)")
        print("="*70)
        for output_len in [128, 256, 512]:
            r = run_bench(
                args.base_url, args.model, "random",
                input_len=128, output_len=output_len,
                num_prompts=10, max_concurrency=1,
                request_rate="inf", ignore_eos=True,
                result_dir=args.result_dir,
                label=f"decode_out{output_len}",
            )
            all_results.append(r)

    # ---- Phase 3: Throughput scaling ----
    # Fixed input/output, varying concurrency (burst all at once)
    if args.phase in ("all", "throughput"):
        print("\n" + "="*70)
        print("PHASE 3: THROUGHPUT SCALING")
        print("="*70)
        # max_num_seqs=8 on current config, so test 1,2,4,8
        for concurrency in [1, 2, 4, 8]:
            r = run_bench(
                args.base_url, args.model, "random",
                input_len=512, output_len=128,
                num_prompts=concurrency * 4,  # 4x the concurrency for stable measurement
                max_concurrency=concurrency,
                request_rate="inf", ignore_eos=True,
                result_dir=args.result_dir,
                label=f"throughput_c{concurrency}",
            )
            all_results.append(r)

    # ---- Phase 4: Realistic mixed workload ----
    if args.phase in ("all", "mixed"):
        print("\n" + "="*70)
        print("PHASE 4: REALISTIC MIXED (512/256, concurrency sweep)")
        print("="*70)
        for concurrency in [1, 4, 8]:
            r = run_bench(
                args.base_url, args.model, "random",
                input_len=512, output_len=256,
                num_prompts=concurrency * 4,
                max_concurrency=concurrency,
                request_rate="inf", ignore_eos=True,
                result_dir=args.result_dir,
                label=f"mixed_512_256_c{concurrency}",
            )
            all_results.append(r)

    # ---- Summary report ----
    summary_file = Path(args.result_dir) / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "model": args.model,
            "base_url": args.base_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": all_results,
        }, f, indent=2)

    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Label':<30} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'Throughput(tok/s)':>18} {'Status':>8}")
    print("-" * 80)
    for r in all_results:
        if r["status"] != "success":
            print(f"{r['label']:<30} {'':>10} {'':>10} {'':>18} {'FAIL':>8}")
            continue
        ttft = r.get("ttft", {}).get("mean", 0) * 1000 if isinstance(r.get("ttft"), dict) else 0
        tpot = r.get("tpot", {}).get("mean", 0) * 1000 if isinstance(r.get("tpot"), dict) else 0
        tput = r.get("total_token_throughput", 0)
        print(f"{r['label']:<30} {ttft:>10.1f} {tpot:>10.1f} {tput:>18.1f} {'OK':>8}")

    print(f"\nFull results saved to: {summary_file}")
    print(f"Individual JSON files in: {args.result_dir}/")


if __name__ == "__main__":
    main()
