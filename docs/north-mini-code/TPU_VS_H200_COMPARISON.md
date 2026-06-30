# NMC TPU v7x vs H200 GPU — Benchmark Comparison

**Date:** 2026-06-30
**Model:** CohereLabs/North-Mini-Code-1.0 (30.48B params, BF16, MoE 128/8)
**Method:** vllm bench serve with synthetic random data, --ignore-eos, identical benchmark script

---

## Hardware Comparison

| Spec | TPU v7x-4 (2x2x1) | H200 (4× GPU) | Ratio (TPU/H200) |
|------|-------------------|---------------|-------------------|
| **Accelerators** | 4 chips | 4 GPUs | 1:1 |
| **HBM capacity** | 95 GB/chip (380 GB total) | 141 GB/GPU (564 GB total) | 0.67× |
| **HBM bandwidth** | 7.4 TB/s/chip (29.6 TB/s total) | 4.8 TB/s/GPU (19.2 TB/s total) | **1.54×** |
| **BF16 compute** | 2,157 TFLOP/s/chip (8,628 total) | ~990 TFLOP/s/GPU (3,956 total) | **2.18×** |
| **Interconnect** | ICI (2x2x1 mesh) | NVLink + NCCL |

## Software Configuration

| Config | TPU v7x | H200 |
|--------|---------|------|
| **vLLM** | 0.23.1rc1 (tpu-inference fork, JAX/Pallas) | vllm/vllm-openai:latest (GPU, CUDA) |
| **TP** | 4 | 4 |
| **max-model-len** | 8192 | 320000 |
| **max-num-seqs** | 32 (tuned) | 256 (vLLM default) |
| **max-num-batched-tokens** | 4096 | 8192 (vLLM default) |
| **dtype** | bfloat16 | bfloat16 |

---

## 1. Single-Stream Decode Latency (TPOT)

*Input=128, output varies, concurrency=1*

| Output Length | TPU TPOT (ms) | H200 TPOT (ms) | TPU / H200 |
|-------------|---------------|----------------|------------|
| 128 | 6.16 | 3.95 | 1.56× |
| 256 | 6.16 | 3.95 | 1.56× |
| 512 | 6.17 | 3.95 | 1.56× |

**Takeaway:** H200 decodes 1.56× faster per-token at batch=1. TPU TPOT is extremely stable (no KV cache growth penalty). The gap is consistent with Python dispatch overhead on the TPU side (P-4 trace: 91% TPU idle at small batch).

---

## 2. Prefill Latency (TTFT)

*Output=1, concurrency=1, post-compile*

| Input Length | TPU TTFT (ms) | H200 TTFT (ms) | TPU / H200 |
|-------------|---------------|----------------|------------|
| 128 | 19.7 | 12.7 | 1.55× |
| 512 | 33.8 | 14.0 | 2.41× |
| 1024 | 42.4 | 16.0 | 2.65× |
| 2048 | 77.2 | 20.7 | 3.73× |
| 4096 | 150.8 | 29.3 | 5.15× |

**Takeaway:** H200 prefill is significantly faster, with the gap widening at longer inputs. At 4096 tokens, H200 is 5.15× faster. The TPU's prefill path has higher per-step dispatch overhead and the RPA v3 kernel has fixed block sizes that don't fully amortize at these lengths.

---

## 3. Throughput Scaling (512→128)

| Concurrency | TPU TPOT (ms) | H200 TPOT (ms) | TPU Output (tok/s) | H200 Output (tok/s) | TPU Total (tok/s) | H200 Total (tok/s) |
|-------------|---------------|----------------|---------------------|---------------------|-------------------|-------------------|
| 1 | 6.16 | 3.95 | 157 | 246 | 783 | 1,228 |
| 2 | 6.53 | 4.36 | 280 | 439 | 1,400 | 2,197 |
| 4 | 6.98 | 4.59 | 531 | 835 | 2,652 | 4,174 |
| 8 | 7.85 | 5.24 | 1,024 | 1,459 | 1,200* | 7,295 |
| 16 | 8.44 | 5.97 | 1,828 | 2,523 | 4,600 | 12,613 |
| 32 | 19.66 | 7.01 | 2,528 | 4,267 | 1,892* | 21,336 |

*TPU c8/c32 total throughput in phase 3 affected by recompile artifacts; see phase 4 for clean numbers.*

---

## 4. Realistic Mixed Workload (512→256) — Clean Comparison

| Concurrency | TPU TTFT (ms) | H200 TTFT (ms) | TPU TPOT (ms) | H200 TPOT (ms) | TPU Output (tok/s) | H200 Output (tok/s) | TPU Total (tok/s) | H200 Total (tok/s) |
|-------------|---------------|----------------|---------------|----------------|---------------------|---------------------|-------------------|-------------------|
| 1 | 34.7 | 18.9 | 6.41 | 3.95 | 153 | 249 | 459 | 747 |
| 4 | 79.3 | 30.1 | 6.96 | 4.60 | 553 | 850 | 1,658 | 2,550 |
| 8 | 113.7 | 36.9 | 7.54 | 5.24 | 1,003 | 1,486 | 3,009 | 4,458 |
| 16 | 194.5 | 50.6 | 8.44 | 5.97 | 1,676 | 2,609 | 5,027 | 7,826 |
| 32 | 186.5 | 62.4 | 13.57 | 7.03 | 2,233 | 4,401 | 6,698 | 13,203 |

### Throughput Ratio (H200 / TPU) at Each Batch Level

| Concurrency | Output Throughput Ratio | Total Throughput Ratio | TPOT Ratio (TPU/H200) |
|-------------|------------------------|------------------------|----------------------|
| 1 | 1.63× | 1.63× | 1.62× |
| 4 | 1.54× | 1.54× | 1.51× |
| 8 | 1.48× | 1.48× | 1.44× |
| 16 | 1.56× | 1.56× | 1.41× |
| 32 | 1.97× | 1.97× | 1.93× |

---

## 5. Scaling Efficiency (relative to batch=1)

| Concurrency | TPU Scaling | H200 Scaling |
|-------------|-------------|--------------|
| 1 | 1.00× (153 tok/s) | 1.00× (249 tok/s) |
| 4 | 3.61× (553 tok/s) | 3.41× (850 tok/s) |
| 8 | 6.55× (1,003 tok/s) | 5.97× (1,486 tok/s) |
| 16 | 10.95× (1,676 tok/s) | 10.48× (2,609 tok/s) |
| 32 | 14.60× (2,233 tok/s) | 17.68× (4,401 tok/s) |

**Takeaway:** Both platforms scale similarly to batch=16 (~10-11×). At batch=32, the H200 continues scaling (17.7×) while the TPU saturates (14.6×) — the TPU hits HBM bandwidth limits and TPOT degrades sharply (13.57ms vs H200's 7.03ms).

---

## Key Findings

### 1. H200 is 1.5–2× faster across all metrics
- **Single-stream decode:** H200 3.95ms/tok vs TPU 6.16ms/tok (1.56×)
- **Batch=32 throughput:** H200 4,401 tok/s vs TPU 2,233 tok/s (1.97×)
- **Prefill TTFT at 4096:** H200 29ms vs TPU 151ms (5.15×)

### 2. The gap is NOT raw hardware capability
The TPU v7x has **1.54× more HBM bandwidth** (29.6 vs 19.2 TB/s) and **2.18× more BF16 compute** (8,628 vs 3,956 TFLOP/s) than 4× H200. Despite this hardware advantage, the TPU is slower. The bottleneck is:
- **Python dispatch overhead** (P-4 trace: 91% TPU idle at small batch)
- **Prefill kernel efficiency** (RPA v3 block sizes not amortizing at long contexts)
- **TP all-reduce overhead** (2 fused collectives/step)

### 3. Scaling efficiency is comparable to batch=16
Both platforms achieve ~10-11× scaling at batch=16. The TPU actually scales proportionally slightly better (10.95× vs 10.48×). The divergence happens at batch=32 where the TPU saturates.

### 4. TPU reaches 56.5% HBM utilization at batch=32
The HBM-bound theoretical for TPU decode is 3,950 tok/s; actual is 2,233 tok/s (56.5%). This is approaching the efficiency ceiling — further gains require reducing dispatch overhead, not more batching.

### 5. Config differences matter
The H200 uses vLLM defaults (max-num-seqs=256, max-num-batched-tokens=8192) while the TPU was tuned to 32/4096. The H200's larger batch capacity gives it an advantage at high concurrency. Matching configs would narrow but not eliminate the gap.

---

## Recommendations for Closing the Gap

1. **Reduce Python dispatch overhead** — the #1 bottleneck. JAX custom-call dispatch adds ~5ms/step on top of <1ms TPU execution time. Options: JAX async dispatch, CUDA-graph-style capture, or reducing Python layers per step.

2. **Tune prefill block sizes** — the RPA v3 kernel's fixed block sizes don't amortize at long contexts. Larger block sizes for prefill could close the TTFT gap (currently 5.15× at 4096).

3. **Increase max-num-seqs beyond 32** — the TPU saturates at batch=32, but with reduced dispatch overhead, higher batch sizes could be viable.

4. **Expert parallelism (EP)** — eliminates MoE all-reduce by sharding experts across devices. Would reduce collective overhead at the cost of increased gather traffic.

5. **JAX compilation improvements** — pre-compile common batch sizes to avoid runtime recompilation stalls.
