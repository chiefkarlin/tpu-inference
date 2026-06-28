#!/bin/bash
# P-4 Trace capture for NMC on v7x.
#
# Captures a JAX profiler trace of NMC inference (prefill + decode) from a
# running vLLM server that was started with --profiler torch.
#
# Prerequisites:
#   - The nmc-v7x-inference pod is Running and healthy
#   - The pod was started with VLLM_ARGS including:
#       --profiler torch --torch-profiler-dir /tmp/profiles
#     (Use nmc-v7x-pod-profiling.yaml which has these args pre-configured)
#   - KUBECONFIG points at the TPU cluster
#
# Usage:
#   ./capture_nmc_trace.sh                # capture + download trace
#   ./capture_nmc_trace.sh --analyze      # capture + download + analyze
#
# Output:
#   /tmp/nmc_trace/  — downloaded xplane.pb trace files
#   (with --analyze) — per-kernel timing report printed to stdout
#
set -euo pipefail

POD_NAME="nmc-v7x-inference"
NAMESPACE="default"
PROFILE_DIR="/tmp/profiles"
LOCAL_TRACE_DIR="/tmp/nmc_trace"
ANALYZE="${1:-}"
MODEL="CohereLabs/North-Mini-Code-1.0"

echo "=== NMC P-4 Trace Capture ==="
echo "  Pod:      ${POD_NAME}"
echo "  Profile:  ${PROFILE_DIR} (in-pod)"
echo "  Output:   ${LOCAL_TRACE_DIR} (local)"
echo ""

# --- 0. Verify pod is running ---
echo "--- 0. Verify pod status ---"
POD_STATUS=$(kubectl get pod "${POD_NAME}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
if [[ "${POD_STATUS}" != "Running" ]]; then
    echo "  ERROR: pod ${POD_NAME} is ${POD_STATUS}, expected Running"
    echo "  Start the pod first (use nmc-v7x-pod-profiling.yaml for profiling-enabled deployment)"
    exit 1
fi
echo "  OK: pod is Running"
echo ""

# --- 1. Health check + warmup ---
echo "--- 1. Health check + warmup ---"
kubectl exec "${POD_NAME}" -- python3 -c "
import urllib.request, json, time, sys

# Wait for health
for i in range(60):
    try:
        urllib.request.urlopen('http://localhost:8000/health', timeout=5)
        print('  Server healthy')
        break
    except:
        if i % 10 == 0:
            print(f'  Waiting for server... ({i}s)')
        time.sleep(5)
else:
    print('  ERROR: server not healthy after 300s')
    sys.exit(1)

# Warmup request (triggers compilation)
print('  Sending warmup request (triggers XLA compile)...')
body = json.dumps({
    'model': '${MODEL}',
    'prompt': 'def hello():',
    'max_tokens': 4,
    'temperature': 0.0,
}).encode()
req = urllib.request.Request(
    'http://localhost:8000/v1/completions',
    data=body,
    headers={'Content-Type': 'application/json'},
    method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=300)
    data = json.loads(resp.read().decode())
    print(f'  Warmup OK: generated {len(data[\"choices\"][0][\"text\"])} chars')
except Exception as e:
    print(f'  Warmup FAILED: {e}')
    sys.exit(1)
"
echo ""

# --- 2. Start profiling ---
echo "--- 2. Start profiling ---"
kubectl exec "${POD_NAME}" -- python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://localhost:8000/start_profile',
    data=b'',
    headers={'Content-Type': 'application/json'},
    method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=30)
    print(f'  start_profile: {resp.status} {resp.read().decode().strip()}')
except Exception as e:
    print(f'  start_profile FAILED: {e}')
    print('  Ensure pod was started with --profiler torch --torch-profiler-dir /tmp/profiles')
    import sys; sys.exit(1)
"
echo ""

# --- 3. Send profiled requests (prefill + decode) ---
echo "--- 3. Sending profiled requests ---"
kubectl exec "${POD_NAME}" -- python3 -c "
import urllib.request, json, time

prompts = [
    'def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    ',
    'Write a Python function to check if a string is a palindrome.',
    'Explain how to implement a binary search tree in Python.',
]

for i, prompt in enumerate(prompts):
    body = json.dumps({
        'model': '${MODEL}',
        'prompt': prompt,
        'max_tokens': 128,
        'temperature': 0.0,
    }).encode()
    req = urllib.request.Request(
        'http://localhost:8000/v1/completions',
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST')
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=300)
    elapsed = time.time() - t0
    data = json.loads(resp.read().decode())
    text = data['choices'][0]['text'][:80]
    tokens = data.get('usage', {}).get('completion_tokens', '?')
    print(f'  Request {i+1}: {tokens} tokens in {elapsed:.2f}s ({float(tokens)/elapsed:.1f} tok/s)')
    print(f'    Output: {text}...')
"
echo ""

# --- 4. Stop profiling ---
echo "--- 4. Stop profiling ---"
kubectl exec "${POD_NAME}" -- python3 -c "
import urllib.request, json, time
# Small delay to ensure last decode step is captured
time.sleep(2)
req = urllib.request.Request(
    'http://localhost:8000/stop_profile',
    data=b'',
    headers={'Content-Type': 'application/json'},
    method='POST')
try:
    resp = urllib.request.urlopen(req, timeout=60)
    print(f'  stop_profile: {resp.status} {resp.read().decode().strip()}')
except Exception as e:
    print(f'  stop_profile FAILED: {e}')
"
echo ""

# --- 5. Download trace ---
echo "--- 5. Download trace ---"
rm -rf "${LOCAL_TRACE_DIR}"
mkdir -p "${LOCAL_TRACE_DIR}"

# List trace files in the pod
echo "  Trace files in pod:"
kubectl exec "${POD_NAME}" -- ls -la "${PROFILE_DIR}" 2>&1 || echo "  (no files found at ${PROFILE_DIR})"

# Copy trace files
echo "  Downloading..."
kubectl cp "${POD_NAME}:${PROFILE_DIR}/." "${LOCAL_TRACE_DIR}/" 2>&1 || {
    echo "  kubectl cp failed, trying individual files..."
    # Try finding xplane files
    TRACE_FILES=$(kubectl exec "${POD_NAME}" -- find "${PROFILE_DIR}" -name "*.xplane.pb" -o -name "*.pb" 2>/dev/null || echo "")
    if [[ -z "${TRACE_FILES}" ]]; then
        echo "  ERROR: no trace files found in pod at ${PROFILE_DIR}"
        exit 1
    fi
    while IFS= read -r f; do
        echo "  Downloading ${f}..."
        kubectl cp "${POD_NAME}:${f}" "${LOCAL_TRACE_DIR}/$(basename ${f})" 2>&1
    done <<< "${TRACE_FILES}"
}

echo "  Downloaded files:"
ls -la "${LOCAL_TRACE_DIR}/" 2>&1
echo ""

# --- 6. Analyze (optional) ---
if [[ "${ANALYZE}" == "--analyze" ]]; then
    echo "--- 6. Analyze trace ---"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze_nmc_trace.py"

    # Find the xplane.pb file
    XPLANE=$(find "${LOCAL_TRACE_DIR}" -name "*.xplane.pb" -o -name "*.pb" | head -1)
    if [[ -z "${XPLANE}" ]]; then
        echo "  ERROR: no .xplane.pb or .pb file found in ${LOCAL_TRACE_DIR}"
        exit 1
    fi

    echo "  Analyzing: ${XPLANE}"
    echo "  (requires: pip install tensorflow pandas)"
    python3 "${ANALYZE_SCRIPT}" "${XPLANE}" 2>&1 || echo "  Analysis failed (install deps first: pip install tensorflow pandas)"
fi

echo ""
echo "=== Trace capture complete ==="
echo "  Trace:    ${LOCAL_TRACE_DIR}/"
echo "  Analyze:  python3 ${ANALYZE_SCRIPT:-analyze_nmc_trace.py} <xplane.pb>"
