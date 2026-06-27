# Execution Runbook: NMC on v7x-4

End-to-end guide for deploying and profiling **CohereLabs/North-Mini-Code-1.0**
on the **robv-tpu-v7x** GKE cluster. Every command is copy-paste executable by
an engineer with GCP access.

**Branch:** `feature/north-mini-code-perf` (HEAD contains the complete NMC
stack: model code + Docker + k8s + profiling tooling).
**Fork:** `https://github.com/chiefkarlin/tpu-inference.git`

---

## 1. Prerequisites

### 1.1 GCP credentials

You need a GCP service-account JSON key with these roles on the TPU cluster's
project (`northam-ce-mlai-tpu`):

| Role | Purpose |
|------|---------|
| `roles/container.developer` | `kubectl get pods`, apply manifests |
| `roles/artifactregistry.writer` | Push Docker image to GAR |
| `roles/cloudbuild.builds.editor` | Submit Cloud Build jobs |
| `roles/tpu.admin` | (optional) Direct TPU admin if debugging |

```bash
# Activate the service account
gcloud auth activate-service-account --key-file=<key.json>
gcloud config set project northam-ce-mlai-tpu

# Configure Docker auth for GAR
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 1.2 HuggingFace token

The `HF_TOKEN` is available in the Scion project environment (export as
`HF_TOKEN` before running the commands below). You'll inject it as a
Kubernetes secret (§4).

### 1.3 Local tools

```bash
# Verify kubectl + gcloud are installed
kubectl version --client
gcloud --version
```

### 1.4 Files in this repo (all on feature/north-mini-code-perf)

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Base image (vLLM + tpu_inference editable install) |
| `examples/nmc/build_nmc_image.sh` | Build + push script (3 modes) |
| `examples/nmc/cloudbuild.yaml` | Cloud Build config (no local Docker needed) |
| `examples/nmc/nmc-v7x-pod.yaml` | Kubernetes pod manifest |
| `examples/nmc/e2e_sanity_check.sh` | Correctness sanity check (curl) |
| `examples/nmc/analyze_nmc_trace.py` | Trace analysis (self-contained, needs tensorflow+pandas) |
| `docs/north-mini-code/PROFILING_PLAN.md` | Profiling strategy + HBM floor analysis |
| `docs/north-mini-code/ANALYSIS_FINDINGS.md` | P-1/P-2 corrected findings |

---

## 2. Cluster Auth

```bash
# Get cluster credentials (writes ~/.kube/config)
gcloud container clusters get-credentials \
    gke_northam-ce-mlai-tpu_us-central1-c_robv-tpu-v7x \
    --region us-central1-c

# Verify access
kubectl get nodes
# Expected: 1+ nodes with tpu7x accelerator, node pool single-host-vllm

kubectl get pods -A
# Expected: system pods running
```

**If this fails with MetadataServerException:** the metadata-server concealment
is blocking gcloud. Use the service-account key path (§1.1) instead of
metadata-server auth. The SA key bypasses the metadata server entirely.

---

## 3. Image Build

### 3.1 Set variables

```bash
GAR_PROJECT=northam-ce-mlai-tpu
GAR_LOCATION=us-central1
GAR_REPO=nmc
IMAGE_TAG=latest
GAR_IMAGE="${GAR_LOCATION}-docker.pkg.dev/${GAR_PROJECT}/${GAR_REPO}/nmc-inference:${IMAGE_TAG}"
```

### 3.2 Option A: Cloud Build (recommended — no local Docker needed)

```bash
cd /workspace/tpu-inference   # repo root with feature/north-mini-code-perf checked out

gcloud builds submit \
    --config examples/nmc/cloudbuild.yaml \
    --substitutions _GAR_IMAGE="${GAR_IMAGE}" \
    --project "${GAR_PROJECT}" \
    .
```

Build takes ~20-40 min (compiles vLLM from source). Cloud Build logs stream
to the terminal. The image is pushed to GAR automatically.

### 3.2 Option B: Local Docker build + push

```bash
cd /workspace/tpu-inference

# Build (pins vLLM to LKG commit from .buildkite/vllm_lkg.version)
GAR_PROJECT=${GAR_PROJECT} ./examples/nmc/build_nmc_image.sh --push

# Or manually:
VLLM_LKG=$(cat .buildkite/vllm_lkg.version)
docker build -f docker/Dockerfile \
    --build-arg VLLM_COMMIT_HASH="${VLLM_LKG}" \
    -t "${GAR_IMAGE}" .
docker push "${GAR_IMAGE}"
```

### 3.3 Verify

```bash
gcloud artifacts docker images list \
    "${GAR_LOCATION}-docker.pkg.dev/${GAR_PROJECT}/${GAR_REPO}" \
    --project "${GAR_PROJECT}"
```

---

## 4. Pod Deploy

### 4.1 Create HF token secret

```bash
kubectl create secret generic hf-token-secret \
    --from-literal=token="${HF_TOKEN}"
```

### 4.2 Update pod manifest with GAR image

```bash
# Replace <GAR_IMAGE> placeholder in nmc-v7x-pod.yaml
sed -i "s|<GAR_IMAGE>|${GAR_IMAGE}|g" examples/nmc/nmc-v7x-pod.yaml
```

### 4.3 Apply

```bash
kubectl apply -f examples/nmc/nmc-v7x-pod.yaml
```

### 4.4 Watch rollout

```bash
# Watch pod status (weight download ~61GB + compile = 5-15 min)
kubectl get pods nmc-v7x-inference -w

# Check readiness (readinessProbe hits /health on port 8000)
kubectl describe pod nmc-v7x-inference | grep -A5 "Readiness"

# Stream logs (watch for "Starting vLLM OpenAI-compatible API server")
kubectl logs nmc-v7x-inference -f

# Key log milestones to watch for:
#   1. "Starting vLLM OpenAI-compatible API server..."
#   2. HF weight download (49 safetensors shards, ~61GB)
#   3. JAX mesh initialization (TP=4, 4 chips)
#   4. XLA compilation (first request triggers compile, ~2-5 min)
#   5. "Application startup complete" → server ready
```

### 4.5 Port-forward for local access

```bash
kubectl port-forward pod/nmc-v7x-inference 8000:8000
# Now http://localhost:8000 is the vLLM API
```

---

## 5. E2E Sanity Check

### 5.1 Run the sanity check script

```bash
# From the repo root (with feature/north-mini-code-perf checked out)
# If port-forwarding:
./examples/nmc/e2e_sanity_check.sh localhost

# Or directly to the pod IP:
POD_IP=$(kubectl get pod nmc-v7x-inference -o jsonpath='{.status.podIP}')
./examples/nmc/e2e_sanity_check.sh "${POD_IP}"
```

### 5.2 Expected output

```
=== NMC E2E Sanity Check ===
--- 1. Health check ---
  OK: server is healthy
--- 2. Models list ---
  { "data": [{ "id": "CohereLabs/North-Mini-Code-1.0", ... }] }
--- 3. Completions API: coding prompt ---
  Generated text: <coherent Python code, e.g. "if n <= 1:\n    return n\n...">
--- 4. Chat API: coding question ---
  (may return 400 if no chat template — use /v1/completions instead)
=== Summary ===
  Health:       OK
  Completions:  OK
  Correctness sanity check passed. Model is generating text.
```

**Note:** NMC's tokenizer_config.json has **no chat_template**, so the
`/v1/chat/completions` endpoint may return a 400. Use `/v1/completions`
for all testing. The sanity check tries both but only requires completions
to pass.

### 5.3 Manual test

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "CohereLabs/North-Mini-Code-1.0",
        "prompt": "def quicksort(arr):\n    ",
        "max_tokens": 64,
        "temperature": 0
    }'
```

---

## 6. Trace Capture (P-4 Profiling)

### 6.1 Approach A: Offline profiling (recommended for first trace)

This uses `examples/tpu_profiling.py` which creates a standalone `LLM`
instance with profiler config set programmatically. Run it in a **separate
pod** (or stop the serving pod first to avoid TPU contention).

```bash
# Option 1: Start a new profiling pod (bash mode, no server)
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: nmc-profiling
  labels:
    app: nmc-profiling
spec:
  nodeSelector:
    cloud.google.com/gke-tpu-accelerator: tpu7x
    cloud.google.com/gke-tpu-topology: 2x2x1
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: cloud.google.com/gke-nodepool
                operator: In
                values: [single-host-vllm]
  restartPolicy: Never
  containers:
    - name: profiler
      image: ${GAR_IMAGE}
      imagePullPolicy: Always
      command: ["/bin/bash", "-c", "sleep infinity"]
      env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        - name: SKIP_JAX_PRECOMPILE
          value: "1"
      resources:
        requests:
          google.com/tpu: 4
        limits:
          google.com/tpu: 4
      volumeMounts:
        - name: hf-cache
          mountPath: /root/.cache/huggingface
        - name: profiles
          mountPath: /tmp/profiles
  volumes:
    - name: hf-cache
      emptyDir:
        sizeLimit: 100Gi
    - name: profiles
      emptyDir:
        sizeLimit: 20Gi
EOF

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/nmc-profiling --timeout=600s
```

```bash
# Run the profiling script inside the pod
kubectl exec -it nmc-profiling -- python3 /workspace/tpu_inference/examples/tpu_profiling.py \
    --model CohereLabs/North-Mini-Code-1.0 \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --trust-remote-code \
    --input-len 1 \
    --output-len 128 \
    --batch-size 1 \
    --num-iters-warmup 3 \
    --num-iters 5 \
    --profile-result-dir /tmp/profiles
```

This:
1. Loads the model (downloads weights if not cached, ~5-10 min)
2. Warms up (3 iterations — triggers XLA compilation)
3. Starts `jax.profiler.start_trace("/tmp/profiles")`
4. Runs 5 profiled decode iterations (128 tokens each)
5. Stops trace (`jax.profiler.stop_trace()`)

### 6.2 Download the trace

```bash
# Find the xplane.pb file
kubectl exec nmc-profiling -- find /tmp/profiles -name "*.xplane.pb" -o -name "*.xplane.pb.gz"

# Copy it out
kubectl cp nmc-profiling:/tmp/profiles ./nmc-profiles

# Find the xplane.pb (may be in a subdirectory with timestamp)
find ./nmc-profiles -name "*.xplane.pb*"
```

### 6.3 Approach B: Serving API profiling

If the server is already running and you want to capture a trace of live
traffic:

```bash
# 1. Start profiling (POST /start_profile — requires profiler pre-configured)
curl -X POST http://localhost:8000/start_profile

# 2. Send requests (e.g., 10 decode requests)
for i in $(seq 1 10); do
    curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"CohereLabs/North-Mini-Code-1.0","prompt":"def hello():","max_tokens":32,"temperature":0}' &
done
wait

# 3. Stop profiling
curl -X POST http://localhost:8000/stop_profile

# 4. Download trace (profile_dir is set via --profiler torch --torch-profiler-dir /tmp/profiles in VLLM_ARGS)
kubectl cp nmc-v7x-inference:/tmp/profiles ./nmc-profiles
```

**Note:** Approach B requires `--profiler torch` and `--torch-profiler-dir /tmp/profiles`
in `VLLM_ARGS` in the pod manifest. To enable, uncomment/add these to
`nmc-v7x-pod.yaml`'s `VLLM_ARGS` env var. The `tpu_worker.py` init checks
`profiler_config.profiler == "torch"` and sets `self.profile_dir`.

### 6.4 Approach C: JAX profiler server (remote capture)

For interactive profiling with xprof / Cloud TPU profiler:

```bash
# Add these env vars to the pod manifest:
#   USE_JAX_PROFILER_SERVER: "true"
#   JAX_PROFILER_SERVER_PORT: "9999"

# Port-forward the profiler port
kubectl port-forward pod/nmc-v7x-inference 9999:9999

# Connect with xprof (from a machine with tensorboard-plugin-profile installed)
# tensorboard --logdir /tmp/profiles --port 6006
# Then use the Profile tab to capture a trace remotely.
```

---

## 7. Trace Analysis

### 7.1 Install dependencies

```bash
# Only two deps needed (analyze_nmc_trace.py is self-contained)
pip install tensorflow pandas
```

### 7.2 Run the analysis

```bash
# Find the xplane.pb file
XPLANE=$(find ./nmc-profiles -name "*.xplane.pb*" | head -1)

# Generate the report
python3 /workspace/tpu-inference/examples/nmc/analyze_nmc_trace.py \
    "${XPLANE}" \
    --output nmc_trace_report.md

# Or print to stdout
python3 /workspace/tpu-inference/examples/nmc/analyze_nmc_trace.py "${XPLANE}"
```

### 7.3 What the report contains

1. **Overview Metrics** — device count, total duration, duty cycle, step time
2. **Compute vs Memory Ratio** — SyncWait fraction over last 2 jit_computation
   events on /device:TPU:0. >50% = memory-bound, <30% = compute-bound.
3. **Top 30 Ops** — by total duration (identifies hotspots)
4. **JIT/Pallas Computations** — per-kernel timings
5. **Attention (RPA) Kernels** — ragged_paged_attention timings (scope name
   includes `RPA{D|P|M}-p_{page_size}-bq_{bq}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}-sw_{sliding_window}`)
6. **MoE GEMM Kernels** — gmm_v2 / megablox timings
7. **SyncWait/DMA Stalls** — memory stall events
8. **HBM-Bound Reference** — theoretical minimum (1.87 GB/chip, 0.253 ms/token)

### 7.4 Interpreting results

| Metric | Target | Action if off |
|--------|--------|---------------|
| Decode step time | ≤ 0.5 ms/token (2× HBM floor) | If >> 0.5ms: check SyncWait ratio, DMA overlap |
| SyncWait ratio (decode) | < 30% (compute-bound) | If > 50%: HBM-bound — increase DMA/compute overlap via block_sizes |
| RPA scope name | `RPA-p_16-bq_1_1-bkv_4096_2048-sw_4096` | If bkv != 4096: d_block_sizes not wired correctly |
| MoE GMM time | < 0.15 ms/layer (48 layers) | If high: check gmm_v2 tiling (calculate_tiling heuristic) |
| Prefill step time (P=1024) | ≤ 5 ms | If high: check prefill block_sizes (default heuristic) |

---

## 8. Troubleshooting

### 8.1 Pod won't start / CrashLoopBackOff

```bash
# Check events
kubectl describe pod nmc-v7x-inference | tail -30

# Common causes:
# - Image not found: verify GAR image path, check imagePullSecrets
# - TPU resources unavailable: check node pool, kubectl get nodes
# - HF_TOKEN invalid: check secret, kubectl get secret hf-token-secret -o yaml
```

### 8.2 Model won't load / compilation error

```bash
# Stream logs
kubectl logs nmc-v7x-inference -f

# Common causes:
# - "UnsupportedArchitectureError": model not registered — verify model_loader.py has Cohere2MoeForCausalLM
# - "AttributeError: moe_intermediate_size": BUG #1 not fixed — verify config.intermediate_size is used
# - "shape mismatch" in weight loading: BUG #2 not fixed — verify prefix_dense_intermediate_size
# - XLA compilation OOM: reduce --max-model-len, --max-num-seqs, --gpu-memory-utilization
# - "sliding_window" not in AttentionMetadata: per-layer injection missing in Model.__call__
```

### 8.3 Server starts but requests fail

```bash
# Check if model compiled
kubectl logs nmc-v7x-inference | grep -i "compile\|error\|traceback"

# Test health
curl http://localhost:8000/health

# Test with verbose error
curl -v http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"CohereLabs/North-Mini-Code-1.0","prompt":"hello","max_tokens":8}'
```

### 8.4 The 4 kernel hotspots for triage

| # | Hotspot | Location | Symptom | Tuning lever |
|---|---------|----------|---------|--------------|
| 1 | **Sliding-window attention decode** | `cohere2_attention.py` → RPA v3 | High SyncWait on sliding layers | `d_block_sizes=(bq,bkv,bq_csz,bkv_csz)` — current: (1,4096,1,2048). If VMEM OK, try (1,8192,1,4096) for fewer DMA launches. If VMEM tight, try (1,2048,1,1024). |
| 2 | **MoE GMM1 (gate+up)** | `gmm_v2` via `fused_moe_gmm.py:209` | High time on GMM1 vs GMM2 | `tile_info` param of gmm_v2 (currently heuristic `calculate_tiling`). Per-chip: size_m=8*topk=64 tokens, size_k=2048, size_n=384 (2F/TP=1536/4). |
| 3 | **MoE GMM2 (down)** | `gmm_v2` via `fused_moe_gmm.py:227` | High time + all-reduce overhead | Same `tile_info`. Per-chip: size_m=64, size_k=192 (F/TP), size_n=2048. Row-parallel → all-reduce at end. |
| 4 | **LM head (tied embed)** | `cohere2_moe.py:608` | Large GEMM: (1, 2048) @ (2048, 65536) | TP-sharded on vocab (P("model",None)). 256 MiB/chip. HBM-bound at 0.035 ms. Little to tune. |

### 8.5 Profiling issues

```bash
# No xplane.pb found after profiling
kubectl exec nmc-profiling -- find /tmp/profiles -type f
# If empty: check that tpu_profiling.py completed without error
# The trace is written on stop_trace() — if the script crashed before that, no trace.

# analyze_nmc_trace.py ImportError
pip install tensorflow pandas
# tensorflow provides tensorflow.tsl.profiler.protobuf.xplane_pb2

# "No jit_computation events found"
# The trace may be from the wrong device. Check:
python3 -c "
from examples.nmc.analyze_nmc_trace import build_sqlite_db
conn = build_sqlite_db('your.xplane.pb')
print(conn.execute('SELECT DISTINCT name FROM planes').fetchall())
"
# Look for /device:TPU:0 in the plane names.
```

### 8.6 Cluster access (metadata concealment)

If `gcloud` crashes with `MetadataServerException`:

```bash
# This means metadata-server concealment is ON in the pod.
# Fix: use a service-account JSON key (NOT metadata-server auth).
gcloud auth activate-service-account --key-file=<key.json>
gcloud config set project northam-ce-mlai-tpu

# Verify
gcloud auth list
gcloud auth print-access-token  # should work now
```

---

## 9. Quick Reference: One-Shot Deployment

```bash
# === SET THESE ===
GAR_PROJECT=northam-ce-mlai-tpu
GAR_IMAGE="us-central1-docker.pkg.dev/${GAR_PROJECT}/nmc/nmc-inference:latest"
# HF_TOKEN must be set in your environment (from the Scion project)

# 1. Auth
gcloud auth activate-service-account --key-file=<key.json>
gcloud container clusters get-credentials gke_northam-ce-mlai-tpu_us-central1-c_robv-tpu-v7x --region us-central1-c
kubectl get nodes  # verify

# 2. Build (Cloud Build, ~30 min)
cd /workspace/tpu-inference
gcloud builds submit --config examples/nmc/cloudbuild.yaml \
    --substitutions _GAR_IMAGE="${GAR_IMAGE}" --project "${GAR_PROJECT}" .

# 3. Deploy
kubectl create secret generic hf-token-secret --from-literal=token=${HF_TOKEN}
sed -i "s|<GAR_IMAGE>|${GAR_IMAGE}|g" examples/nmc/nmc-v7x-pod.yaml
kubectl apply -f examples/nmc/nmc-v7x-pod.yaml

# 4. Wait for readiness (~10-15 min for weight download + compile)
kubectl get pod nmc-v7x-inference -w

# 5. Sanity check
kubectl port-forward pod/nmc-v7x-inference 8000:8000 &
./examples/nmc/e2e_sanity_check.sh localhost

# 6. Profile (separate pod)
# ... see §6.1

# 7. Analyze
pip install tensorflow pandas
python3 examples/nmc/analyze_nmc_trace.py <xplane.pb> --output report.md
```

---

## 10. Architecture Context (for the executing engineer)

- **Model:** Cohere2MoeForCausalLM, 30.48B params BF16, ~3B active
- **Topology:** v7x-4 (4 chips, 2x2x1), 2 cores/chip = 8 cores total
- **TP=4:** 1 JAX process/chip, 2 cores each. NMC has 4 kv_heads → 1 kv_head/process.
- **MoE:** 128 experts, 8 active/token, intermediate=768, hidden=2048, gated SiLU,
  sigmoid routing. Default backend: GMM_TP (tensor-parallel grouped GEMM via gmm_v2).
- **Attention:** GQA 32q/4kv, head_dim=128, sliding_window=4096 (sliding layers) +
  full (full layers). Hybrid schedule: [full,sliding,sliding,sliding]×12 + final full.
  RoPE theta=50000 (interleaved/NeoX) on sliding+dense0 only.
- **Block:** Parallel (x+attn(norm(x))+mlp(norm(x))), single shared RMSNorm.
- **v7x specs:** HBM BW 7.4 TB/s/chip, MXU 2,157 TFLOP/s BF16, HBM 192 GiB/chip.
- **HBM floor:** 1.87 GB/chip decode → 0.253 ms/token theoretical minimum.

For full analysis details, see `docs/north-mini-code/PROFILING_PLAN.md` and
`docs/north-mini-code/ANALYSIS_FINDINGS.md`.
