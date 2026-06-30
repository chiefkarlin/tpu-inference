#!/usr/bin/env python3
"""
Lightweight NMC benchmark — uses direct async HTTP requests instead of vllm bench serve.
Designed to run inside the vLLM pod without OOMing (low memory footprint).
"""
import asyncio
import json
import time
import urllib.request
import sys
import os

BASE_URL = "http://localhost:8000"
MODEL = "CohereLabs/North-Mini-Code-1.0"


def send_request(prompt_tokens, max_tokens, temperature=0):
    """Send a single completion request and return timing data."""
    import random
    # Generate random token IDs as prompt (simulates synthetic data)
    prompt = " ".join(["hello"] * prompt_tokens)

    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": True,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    ttft = time.time() - start  # Approximate TTFT (time to first response byte)
    result = json.loads(resp.read())
    total_time = time.time() - start

    return {
        "ttft": ttft * 1000,  # ms
        "total_time": total_time * 1000,  # ms
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "completion_tokens": result["usage"]["completion_tokens"],
        "tpot": (total_time - ttft) / max(result["usage"]["completion_tokens"], 1) * 1000,  # ms
        "output_tput": result["usage"]["completion_tokens"] / total_time,  # tok/s
    }


def run_sequential(label, input_len, output_len, num_requests=5):
    """Run requests sequentially and report stats."""
    print(f"\n{'='*60}")
    print(f"BENCH: {label} (in={input_len}, out={output_len}, n={num_requests})")
    print(f"{'='*60}")

    # Warmup
    try:
        send_request(16, 1)
    except:
        pass

    results = []
    for i in range(num_requests):
        try:
            r = send_request(input_len, output_len)
            results.append(r)
            print(f"  [{i+1}/{num_requests}] TTFT={r['ttft']:.1f}ms TPOT={r['tpot']:.1f}ms tput={r['output_tput']:.1f}tok/s")
        except Exception as e:
            print(f"  [{i+1}/{num_requests}] FAILED: {e}")

    if not results:
        return {"label": label, "status": "failed"}

    ttfts = [r["ttft"] for r in results]
    tpots = [r["tpot"] for r in results]
    tputs = [r["output_tput"] for r in results]

    ttfts.sort()
    tpots.sort()

    stats = {
        "label": label,
        "status": "success",
        "ttft_p50": ttfts[len(ttfts)//2],
        "tpot_p50": tpots[len(tpots)//2],
        "output_tput": sum(tputs)/len(tputs),
        "n": len(results),
    }
    print(f"  SUMMARY: TTFT p50={stats['ttft_p50']:.1f}ms TPOT p50={stats['tpot_p50']:.1f}ms tput={stats['output_tput']:.1f}tok/s")
    return stats


def run_concurrent(label, input_len, output_len, concurrency, num_requests=None):
    """Run concurrent requests and report aggregate throughput."""
    if num_requests is None:
        num_requests = concurrency * 4

    print(f"\n{'='*60}")
    print(f"BENCH: {label} (in={input_len}, out={output_len}, c={concurrency}, n={num_requests})")
    print(f"{'='*60}")

    import concurrent.futures

    # Warmup
    try:
        send_request(16, 1)
    except:
        pass

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_request, input_len, output_len) for _ in range(num_requests)]
        results = []
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  FAILED: {e}")
    elapsed = time.time() - start

    if not results:
        return {"label": label, "status": "failed"}

    total_output_tokens = sum(r["completion_tokens"] for r in results)
    total_tokens = sum(r["prompt_tokens"] + r["completion_tokens"] for r in results)
    tpots = sorted([r["tpot"] for r in results])
    ttfts = sorted([r["ttft"] for r in results])

    stats = {
        "label": label,
        "status": "success",
        "concurrency": concurrency,
        "n": len(results),
        "elapsed_s": elapsed,
        "ttft_p50": ttfts[len(ttfts)//2],
        "tpot_p50": tpots[len(tpots)//2],
        "output_tput": total_output_tokens / elapsed,
        "total_tput": total_tokens / elapsed,
    }
    print(f"  SUMMARY: TTFT p50={stats['ttft_p50']:.1f}ms TPOT p50={stats['tpot_p50']:.1f}ms "
          f"output={stats['output_tput']:.1f}tok/s total={stats['total_tput']:.1f}tok/s")
    return stats


def main():
    all_results = []

    # Phase 1: Prefill TTFT (sequential)
    print("\n" + "="*60)
    print("PHASE 1: PREFILL LATENCY (TTFT)")
    print("="*60)
    for input_len in [128, 512, 1024, 2048, 4096]:
        r = run_sequential(f"prefill_in{input_len}", input_len, 1, num_requests=5)
        all_results.append(r)

    # Phase 2: Decode TPOT (sequential)
    print("\n" + "="*60)
    print("PHASE 2: DECODE LATENCY (TPOT)")
    print("="*60)
    for output_len in [128, 256, 512]:
        r = run_sequential(f"decode_out{output_len}", 128, output_len, num_requests=5)
        all_results.append(r)

    # Phase 3: Throughput scaling (concurrent)
    print("\n" + "="*60)
    print("PHASE 3: THROUGHPUT SCALING")
    print("="*60)
    for c in [1, 2, 4, 8, 16, 32]:
        r = run_concurrent(f"throughput_c{c}", 512, 128, c)
        all_results.append(r)

    # Phase 4: Realistic mixed (concurrent)
    print("\n" + "="*60)
    print("PHASE 4: REALISTIC MIXED (512→256)")
    print("="*60)
    for c in [1, 4, 8, 16, 32]:
        r = run_concurrent(f"mixed_512_256_c{c}", 512, 256, c)
        all_results.append(r)

    # Summary
    print("\n" + "="*60)
    print("FULL SUMMARY")
    print("="*60)
    print(f"{'Label':<30} {'TTFT p50':>10} {'TPOT p50':>10} {'Output tput':>12} {'Total tput':>12}")
    print("-" * 78)
    for r in all_results:
        if r["status"] != "success":
            print(f"{r['label']:<30} {'FAIL':>10}")
            continue
        ttft = r.get("ttft_p50", 0)
        tpot = r.get("tpot_p50", 0)
        otput = r.get("output_tput", 0)
        ttput = r.get("total_tput", otput)
        print(f"{r['label']:<30} {ttft:>9.1f}ms {tpot:>9.1f}ms {otput:>11.1f} {ttput:>11.1f}")

    # Save JSON
    with open("/tmp/nmc-bench-results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to /tmp/nmc-bench-results.json")


if __name__ == "__main__":
    main()
