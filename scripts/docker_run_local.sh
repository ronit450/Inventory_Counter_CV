#!/usr/bin/env bash
# Run the inventory-counter container locally for testing.
# Usage:
#   VIDEO_URL=s3://bucket/video.mp4 ./scripts/docker_run_local.sh
#   VIDEO_URL=/path/to/local.mp4    ./scripts/docker_run_local.sh
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-inventory-counter}
IMAGE_TAG=${IMAGE_TAG:-latest}

VIDEO_URL=${VIDEO_URL:-}
RESULTS_S3_BUCKET=${RESULTS_S3_BUCKET:-}
OUTPUT_FILE=${OUTPUT_FILE:-/tmp/results.json}

run_args=(
    --rm
    -e NO_DISPLAY=1
    -e OUTPUT_FILE="$OUTPUT_FILE"
)

if [[ -n "$VIDEO_URL" ]]; then
    run_args+=(-e VIDEO_URL="$VIDEO_URL")
fi

if [[ -n "$RESULTS_S3_BUCKET" ]]; then
    run_args+=(-e RESULTS_S3_BUCKET="$RESULTS_S3_BUCKET")
fi

# Forward AWS credentials from host
if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]]; then
    run_args+=(
        -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
        -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
        -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
    )
else
    # Mount ~/.aws so the container can use host profiles
    run_args+=(-v "$HOME/.aws:/root/.aws:ro")
fi

docker run "${run_args[@]}" "$IMAGE_NAME:$IMAGE_TAG"
