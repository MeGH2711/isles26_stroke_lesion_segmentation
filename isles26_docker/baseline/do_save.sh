#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26-baseline-algorithm"
OUTPUT_FILE="$SCRIPT_DIR/${DOCKER_IMAGE_TAG}.tar.gz"

echo "Exporting Docker image '$DOCKER_IMAGE_TAG' to '$OUTPUT_FILE' ..."
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$OUTPUT_FILE"
echo "[OK] Saved $OUTPUT_FILE"
