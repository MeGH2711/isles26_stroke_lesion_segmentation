#!/usr/bin/env bash
#
# Builds the algorithm's Docker image, boots it as an HTTP server that
# implements Grand Challenge's "invoke" API, then exercises it against the
# configured interface. This script:
#   1. Stages the interface's input files into the container's /input mount
#   2. Calls POST /invoke and checks for a success response (HTTP 201 Created)
#   3. Copies whatever the container wrote to /output back to the host

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26-ensemble-algorithm-v2"
CONTAINER_NAME="isles26-ensemble-algorithm-v2_container"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

# Staging directories are bind-mounted into the container as /input and /output
STAGING_INPUT_DIR="${SCRIPT_DIR}/test/.staging_input"
STAGING_OUTPUT_DIR="${SCRIPT_DIR}/test/.staging_output"

# How long to wait for the container's /health endpoint to come up
HEALTH_CHECK_MAX_ATTEMPTS=60
HEALTH_CHECK_DELAY_SECONDS=10
HEALTH_CHECK_TIMEOUT_SECONDS=10

# How long a single /invoke call is allowed to run
INVOKE_TIMEOUT_SECONDS=600

# --- Globals set by setup() -------------------------------------------------
LOG_LINES_SHOWN=0
DOCKER_VOLUME_TAG=""
DOCKER_NETWORK_TAG=""
TESTER_NAME=""
CONTAINER_PORT=4743
BASE_URL=""
GPU_ARGS=""
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    setup
    trap cleanup EXIT

    build_container
    start_container

    # Poll /health until the server signals it's ready
    check_health

    # Copy this interface's input files into the container
    provision "interf0"
    # Call POST /invoke and wait for inference completion
    invoke
    # Copy the results back to the host
    collect_output "interf0"

    log "=== Test run completed successfully! ==="
    log "Outputs are in: ${OUTPUT_DIR}/interf0/"
    log "Save this image for uploading via ./do_save.sh"
}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

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

setup() {
    log "Setup ..."

    # Allow the Docker user to read these on the host
    chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model" 2>/dev/null || true

    # Disable promotional logs from Docker
    export DOCKER_CLI_HINTS=false

    # Detect whether the NVIDIA GPU runtime is available
    if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log "GPU runtime detected — using --gpus all"
        GPU_ARGS="--gpus all"
    else
        log "No GPU runtime detected — running on CPU" "warning"
        GPU_ARGS=""
    fi

    # Create unique tags for this run
    DOCKER_VOLUME_TAG="${DOCKER_IMAGE_TAG}_volume_$(date +%s)"
    DOCKER_NETWORK_TAG="${DOCKER_IMAGE_TAG}_network_$(date +%s)"
    TESTER_NAME="${DOCKER_IMAGE_TAG}_tester_$(date +%s)"

    # Create the Docker network
    docker network create "$DOCKER_NETWORK_TAG" > /dev/null

    # Clean staging dirs
    rm -rf "$STAGING_INPUT_DIR" "$STAGING_OUTPUT_DIR"
    mkdir -p "$STAGING_INPUT_DIR" "$STAGING_OUTPUT_DIR"
}

build_container() {
    log "Building Docker image: $DOCKER_IMAGE_TAG"
    docker build \
        --platform=linux/amd64 \
        --tag "$DOCKER_IMAGE_TAG" \
        "$SCRIPT_DIR"
}

start_container() {
    log "Starting container: $CONTAINER_NAME"

    # Remove any leftover container
    docker rm -f "$CONTAINER_NAME" &> /dev/null || true

    docker run \
        --rm \
        -d \
        --name "$CONTAINER_NAME" \
        --network "$DOCKER_NETWORK_TAG" \
        $GPU_ARGS \
        --platform linux/amd64 \
        -v "${STAGING_INPUT_DIR}:/input:ro" \
        -v "${STAGING_OUTPUT_DIR}:/output" \
        -v "${SCRIPT_DIR}/model:/opt/ml/model:ro" \
        "$DOCKER_IMAGE_TAG"

    # Get the container's IP on the custom network
    CONTAINER_IP=$(docker inspect -f "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" "$CONTAINER_NAME")
    BASE_URL="http://${CONTAINER_IP}:${CONTAINER_PORT}"
    log "Container IP: $CONTAINER_IP, base URL: $BASE_URL"
}

check_health() {
    log "Waiting for /health to return 200 ..."
    local attempt=0
    while (( attempt < HEALTH_CHECK_MAX_ATTEMPTS )); do
        attempt=$((attempt + 1))

        # Show new container logs
        show_new_container_logs

        local status_code
        status_code=$(docker run --rm --network "$DOCKER_NETWORK_TAG" \
            --name "$TESTER_NAME" \
            curlimages/curl:latest \
            --silent --output /dev/null --write-out "%{http_code}" \
            --max-time "$HEALTH_CHECK_TIMEOUT_SECONDS" \
            "${BASE_URL}/health" 2>/dev/null || echo "000")

        if [[ "$status_code" == "200" ]]; then
            log "/health returned 200 — server is ready"
            return 0
        fi

        log "Attempt $attempt/$HEALTH_CHECK_MAX_ATTEMPTS — /health returned $status_code. Retrying in ${HEALTH_CHECK_DELAY_SECONDS}s ..." "warning"
        sleep "$HEALTH_CHECK_DELAY_SECONDS"
    done

    log "ERROR: /health did not return 200 within $((HEALTH_CHECK_MAX_ATTEMPTS * HEALTH_CHECK_DELAY_SECONDS))s" "error"
    show_new_container_logs
    exit 1
}

provision() {
    local interface="$1"
    log "Provisioning input for interface: $interface"

    # Clean and re-populate staging input
    rm -rf "${STAGING_INPUT_DIR:?}/"*
    cp -rL "${INPUT_DIR}/${interface}/"* "$STAGING_INPUT_DIR/"

    # Clean staging output
    rm -rf "${STAGING_OUTPUT_DIR:?}/"*

    log "Input files staged:"
    find "$STAGING_INPUT_DIR" -type f | head -20
}

invoke() {
    log "Calling POST /invoke (timeout: ${INVOKE_TIMEOUT_SECONDS}s) ..."

    local status_code
    status_code=$(docker run --rm --network "$DOCKER_NETWORK_TAG" \
        --name "$TESTER_NAME" \
        curlimages/curl:latest \
        --silent --output /dev/null --write-out "%{http_code}" \
        --max-time "$INVOKE_TIMEOUT_SECONDS" \
        --request POST \
        "${BASE_URL}/invoke" 2>/dev/null || echo "000")

    show_new_container_logs

    if [[ "$status_code" == "201" ]]; then
        log "/invoke returned 201 — inference succeeded"
    else
        log "ERROR: /invoke returned $status_code (expected 201)" "error"
        exit 1
    fi
}

collect_output() {
    local interface="$1"
    local dest="${OUTPUT_DIR}/${interface}"

    log "Collecting output for interface: $interface"

    rm -rf "$dest"
    mkdir -p "$dest"
    cp -r "${STAGING_OUTPUT_DIR}/"* "$dest/" 2>/dev/null || true

    log "Output files:"
    find "$dest" -type f
}

show_new_container_logs() {
    local all_lines
    all_lines=$(docker logs "$CONTAINER_NAME" 2>&1 | wc -l)
    local new_lines=$((all_lines - LOG_LINES_SHOWN))
    if (( new_lines > 0 )); then
        docker logs "$CONTAINER_NAME" 2>&1 | tail -n "$new_lines"
        LOG_LINES_SHOWN=$all_lines
    fi
}

cleanup() {
    log "Cleaning up ..."
    docker rm -f "$CONTAINER_NAME" &> /dev/null || true
    docker network rm "$DOCKER_NETWORK_TAG" &> /dev/null || true
    rm -rf "$STAGING_INPUT_DIR" "$STAGING_OUTPUT_DIR"
}

main "$@"
