#!/usr/bin/env bash
#
# Saves the algorithm's Docker image as a tarball for Grand Challenge.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26-ensemble-algorithm-v3"

export DOCKER_CLI_HINTS=false

log() {
    local message="$1"
    local level="${2:-info}"
    if [[ -t 1 ]]; then
        case "$level" in
            info)    printf "\e[38;2;36;150;237m> %s\e[0m\n" "$message" ;;
            warning) printf "\e[38;2;255;200;0m> %s\e[0m\n" "$message" ;;
            error)   printf "\e[38;2;255;50;50m> %s\e[0m\n" "$message" ;;
            *)       printf "\e[38;2;36;150;237m> %s\e[0m\n" "$message" ;;
        esac
    else
        printf "%s\n" "$message"
    fi
}

log "Saving Docker image: $DOCKER_IMAGE_TAG"

DOCKER_IMAGE_FILE="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}.tar.gz"

docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$DOCKER_IMAGE_FILE"

log "Docker image saved to: $DOCKER_IMAGE_FILE"
log ""
log "=== Next steps ==="
log "1. Upload Docker image tarball to Grand Challenge:"
log "   $DOCKER_IMAGE_FILE"
log ""
log "2. Upload model weights (model.tar.gz) via: Your algorithm > Models"
log ""
log "3. Submit to Preliminary Evaluation phase first for verification"
