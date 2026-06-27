#!/bin/bash
# Build and (optionally) push the NMC inference Docker image.
#
# Usage:
#   ./build_nmc_image.sh                    # build only (local Docker)
#   ./build_nmc_image.sh --push             # build + push to GAR (local Docker)
#   ./build_nmc_image.sh --cloudbuild       # build + push via Cloud Build (no Docker needed)
#
# Environment variables:
#   GAR_LOCATION   - GAR location (default: us-central1)
#   GAR_PROJECT    - GCP project ID (required for --push / --cloudbuild)
#   GAR_REPO       - GAR repo name (default: nmc)
#   IMAGE_TAG      - image tag (default: latest)
#
# The image is built from docker/Dockerfile with:
#   - VLLM_COMMIT_HASH pinned to the repo's LKG (.buildkite/vllm_lkg.version)
#   - The current tpu-inference source (including NMC model code)
#
# NOTE: If Docker is not available (e.g. in a mgmt pod), use --cloudbuild which
# delegates to Google Cloud Build. This requires gcloud auth to be configured
# (e.g. via `gcloud auth activate-service-account --key-file=<key.json>`).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Configuration ---
VLLM_LKG="$(cat "${REPO_ROOT}/.buildkite/vllm_lkg.version")"
GAR_LOCATION="${GAR_LOCATION:-us-central1}"
GAR_REPO="${GAR_REPO:-nmc}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
GAR_PROJECT="${GAR_PROJECT:-}"

LOCAL_IMAGE="nmc-inference:${IMAGE_TAG}"
MODE="${1:-local}"

# --- Cloud Build path (no Docker daemon needed) ---
if [[ "${MODE}" == "--cloudbuild" ]]; then
    if [[ -z "${GAR_PROJECT}" ]]; then
        echo "ERROR: GAR_PROJECT env var must be set for --cloudbuild" >&2
        exit 1
    fi
    GAR_IMAGE="${GAR_LOCATION}-docker.pkg.dev/${GAR_PROJECT}/${GAR_REPO}/nmc-inference:${IMAGE_TAG}"
    echo "=== Cloud Build: ${GAR_IMAGE} ==="
    echo "  vLLM LKG commit: ${VLLM_LKG}"
    echo "  Source: ${REPO_ROOT}"

    cd "${REPO_ROOT}"
    gcloud builds submit \
        --config examples/nmc/cloudbuild.yaml \
        --substitutions _GAR_IMAGE="${GAR_IMAGE}" \
        --project "${GAR_PROJECT}" \
        .

    echo "=== Pushed: ${GAR_IMAGE} ==="
    echo ""
    echo "Update nmc-v7x-pod.yaml image field to: ${GAR_IMAGE}"
    exit 0
fi

# --- Local Docker build path ---
if [[ "${MODE}" == "--push" ]]; then
    if [[ -z "${GAR_PROJECT}" ]]; then
        echo "ERROR: GAR_PROJECT env var must be set for --push" >&2
        exit 1
    fi
    GAR_IMAGE="${GAR_LOCATION}-docker.pkg.dev/${GAR_PROJECT}/${GAR_REPO}/nmc-inference:${IMAGE_TAG}"
    echo "=== Building ${GAR_IMAGE} ==="
else
    GAR_IMAGE=""
    echo "=== Building local image ${LOCAL_IMAGE} (no push) ==="
fi

echo "  vLLM LKG commit: ${VLLM_LKG}"
echo "  Source: ${REPO_ROOT}"

# --- Build ---
cd "${REPO_ROOT}"

docker build \
    -f docker/Dockerfile \
    --build-arg VLLM_COMMIT_HASH="${VLLM_LKG}" \
    -t "${LOCAL_IMAGE}" \
    .

if [[ -n "${GAR_IMAGE}" ]]; then
    echo "=== Tagging as ${GAR_IMAGE} ==="
    docker tag "${LOCAL_IMAGE}" "${GAR_IMAGE}"

    echo "=== Configuring Docker auth for GAR ==="
    gcloud auth configure-docker "${GAR_LOCATION}-docker.pkg.dev"

    echo "=== Pushing to GAR ==="
    docker push "${GAR_IMAGE}"
    echo "=== Pushed: ${GAR_IMAGE} ==="
    echo ""
    echo "Update nmc-v7x-pod.yaml image field to: ${GAR_IMAGE}"
else
    echo ""
    echo "=== Built locally: ${LOCAL_IMAGE} ==="
    echo "To push to GAR: GAR_PROJECT=<project> $0 --push"
    echo "Or use Cloud Build: GAR_PROJECT=<project> $0 --cloudbuild"
fi
