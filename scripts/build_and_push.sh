#!/usr/bin/env bash
# Build image and push to ECR in one step.
# Usage: ./scripts/build_and_push.sh [tag]
set -euo pipefail

IMAGE_TAG=${1:-${IMAGE_TAG:-latest}}
export IMAGE_TAG

echo "=== Building ==="
./scripts/docker_build.sh "$IMAGE_TAG"

echo ""
echo "=== Pushing to ECR ==="
./scripts/ecr_push.sh

echo ""
echo "=== Done: inventory-counter:$IMAGE_TAG pushed to ECR ==="
