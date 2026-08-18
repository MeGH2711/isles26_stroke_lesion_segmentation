#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26-baseline-algorithm"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

docker build \
  --platform=linux/amd64 \
  --build-arg "BUILD_DATE=${BUILD_DATE}" \
  --tag "$DOCKER_IMAGE_TAG" \
  "$SCRIPT_DIR"
