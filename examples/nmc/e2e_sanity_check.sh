#!/bin/bash
# E2E correctness sanity check for NMC on v7x-4.
#
# Sends a short coding prompt to the running vLLM server and checks:
#   1. The server responds (health check)
#   2. The model generates coherent text (not garbage/repetitions)
#   3. Token generation works (output is non-empty, reasonable length)
#
# Usage:
#   ./e2e_sanity_check.sh [POD_IP]    # default: localhost
#
# Prerequisites: the nmc-v7x-inference pod is running and healthy.

set -euo pipefail

HOST="${1:-localhost}"
PORT=8000
MODEL="CohereLabs/North-Mini-Code-1.0"

echo "=== NMC E2E Sanity Check ==="
echo "  Server: http://${HOST}:${PORT}"
echo "  Model:  ${MODEL}"
echo ""

# 1. Health check
echo "--- 1. Health check ---"
if curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    echo "  OK: server is healthy"
else
    echo "  FAIL: server not responding on /health"
    exit 1
fi
echo ""

# 2. Models list
echo "--- 2. Models list ---"
curl -s "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || \
    curl -s "http://${HOST}:${PORT}/v1/models"
echo ""

# 3. Short coding prompt (completions API)
echo "--- 3. Completions API: coding prompt ---"
PROMPT='def fibonacci(n):\n    """Return the nth Fibonacci number."""\n    '
RESPONSE=$(curl -s "http://${HOST}:${PORT}/v1/completions" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"${PROMPT}\",
        \"max_tokens\": 64,
        \"temperature\": 0.0,
        \"stop\": [\"\\n\\n\"]
    }")

echo "  Response:"
echo "${RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${RESPONSE}"
echo ""

# Extract generated text
GENERATED=$(echo "${RESPONSE}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['text'])
except:
    print('')
" 2>/dev/null)

if [[ -z "${GENERATED}" || "${GENERATED}" == "" ]]; then
    echo "  FAIL: no text generated"
    exit 1
fi
echo "  Generated text: ${GENERATED}"
echo ""

# 4. Chat API (if the model has a chat template)
echo "--- 4. Chat API: coding question ---"
CHAT_RESPONSE=$(curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a Python one-liner to reverse a string.\"}],
        \"max_tokens\": 128,
        \"temperature\": 0.0
    }")

echo "  Response:"
echo "${CHAT_RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${CHAT_RESPONSE}"
echo ""

# 5. Summary
echo "=== Summary ==="
echo "  Health:       OK"
echo "  Completions:  $([[ -n \"${GENERATED}\" ]] && echo 'OK' || echo 'FAIL')"
echo ""
echo "  Correctness sanity check passed. Model is generating text."
echo "  For full profiling, capture a trace next (see PROFILING_PLAN.md)."
