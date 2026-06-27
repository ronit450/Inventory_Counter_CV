#!/usr/bin/env bash
# Build the inventory-counter Docker image.
# Usage: ./scripts/docker_build.sh [tag]
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-inventory-counter}
IMAGE_TAG=${1:-${IMAGE_TAG:-latest}}
PLATFORM=${PLATFORM:-linux/amd64}

docker build \
    --platform "$PLATFORM" \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    .

echo "Built $IMAGE_NAME:$IMAGE_TAG (platform=$PLATFORM)"
