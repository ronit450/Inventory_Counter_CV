#!/usr/bin/env bash
# Push the local Docker image to AWS ECR.
# Usage: IMAGE_TAG=v1.2 ./scripts/ecr_push.sh
set -euo pipefail

AWS_REGION=${AWS_REGION:-us-east-1}
REPO_NAME=${REPO_NAME:-inventory-counter}
IMAGE_TAG=${IMAGE_TAG:-latest}
IMAGE_NAME=${IMAGE_NAME:-$REPO_NAME}
ECR_ASSUME_EXISTS=${ECR_ASSUME_EXISTS:-0}

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

if [[ "$ECR_ASSUME_EXISTS" != "1" ]]; then
    if ! aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
        echo "Creating ECR repo: $REPO_NAME"
        aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION" >/dev/null || \
            echo "Warning: could not create repo — assuming it exists."
    fi
fi

aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin \
        "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ECR_URI"
docker push "$ECR_URI"

echo "Pushed: $ECR_URI"
