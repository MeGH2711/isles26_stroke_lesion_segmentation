#!/usr/bin/env bash
#
# Builds the algorithm's Docker image, boots it as an HTTP server that
# implements Grand Challenge's "invoke" API, then exercises it against the
# configured interface.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26-ensemble-algorithm-v3"
CONTAINER_NAME="isles26-ensemble-algorithm-v3_container"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

STAGING_INPUT_DIR="${SCRIPT_DIR}/test/.staging_input"
STAGING_OUTPUT_DIR="${SCRIPT_DIR}/test/.staging_output"

HEALTH_CHECK_MAX_ATTEMPTS=60
HEALTH_CHECK_DELAY_SECONDS=10
HEALTH_CHECK_TIMEOUT_SECONDS=10
INVOKE_TIMEOUT_SECONDS=600

LOG_LINES_SHOWN=0
DOCKER_VOLUME_TAG=""
DOCKER_NETWORK_TAG=""
TESTER_NAME=""
CONTAINER_PORT=4743
BASE_URL=""
GPU_ARGS=""

main() {
    setup
    trap cleanup EXIT

    build_container
    start_container

    check_health

    provision "interf0"
    invoke
    collect_output "interf0"

    log "=== Test run completed successfully! ==="
    log "Outputs are in: ${OUTPUT_DIR}/interf0/"
    log "Save this image for uploading via ./do_save.sh"
}

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

    chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model" 2>/dev/null || true
    export DOCKER_CLI_HINTS=false

    if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log "GPU runtime detected — using --gpus all"
        GPU_ARGS="--gpus all"
    else
        log "No GPU runtime detected — running on CPU" "warning"
        GPU_ARGS=""
    fi

    local random_id
    random_id=$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')
    DOCKER_VOLUME_TAG="${DOCKER_IMAGE_TAG}_volume_${random_id}"
    DOCKER_NETWORK_TAG="${DOCKER_IMAGE_TAG}_network_${random_id}"
    TESTER_NAME="${DOCKER_IMAGE_TAG}_tester_${random_id}"

    docker volume create "$DOCKER_VOLUME_TAG" > /dev/null
    docker network create "$DOCKER_NETWORK_TAG" > /dev/null

    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    chmod -f o+rwx "$OUTPUT_DIR" 2>/dev/null || true
}

cleanup() {
    log "Cleaning up ..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$TESTER_NAME" 2>/dev/null || true
    docker volume rm -f "$DOCKER_VOLUME_TAG" 2>/dev/null || true
    docker network rm -f "$DOCKER_NETWORK_TAG" 2>/dev/null || true
    rm -rf "$STAGING_INPUT_DIR" "$STAGING_OUTPUT_DIR" 2>/dev/null || true
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

    local container_id
    container_id=$(docker run \
        --platform=linux/amd64 \
        --rm \
        --detach \
        --name "$CONTAINER_NAME" \
        --network "$DOCKER_NETWORK_TAG" \
        --memory 32g \
        $GPU_ARGS \
        --volume "$DOCKER_VOLUME_TAG":/output \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG")

    echo "$container_id"

    local container_ip
    container_ip=$(docker inspect \
        --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
        "$CONTAINER_NAME")

    BASE_URL="http://${container_ip}:${CONTAINER_PORT}"
    log "Container IP: $container_ip, base URL: $BASE_URL"
}

show_new_logs() {
    local all_logs
    all_logs=$(docker logs "$CONTAINER_NAME" 2>&1) || true
    local total_lines
    total_lines=$(echo "$all_logs" | wc -l)
    if [[ $total_lines -gt $LOG_LINES_SHOWN ]]; then
        local new_lines=$(( total_lines - LOG_LINES_SHOWN ))
        echo "$all_logs" | tail -n "$new_lines"
        LOG_LINES_SHOWN=$total_lines
    fi
}

check_health() {
    log "Waiting for /health to return 200 ..."
    local url="${BASE_URL}/health"
    local attempt=1

    while [[ $attempt -le $HEALTH_CHECK_MAX_ATTEMPTS ]]; do
        local http_code
        http_code=$(docker run \
            --platform=linux/amd64 \
            --rm \
            --network "$DOCKER_NETWORK_TAG" \
            curlimages/curl:8.10.1 \
            --silent \
            --output /dev/null \
            --write-out "%{http_code}" \
            --max-time "$HEALTH_CHECK_TIMEOUT_SECONDS" \
            "$url" 2>/dev/null) || http_code="000"

        show_new_logs

        if [[ "$http_code" == "200" ]]; then
            log "/health returned 200 — server is ready"
            return 0
        fi

        log "Attempt $attempt/$HEALTH_CHECK_MAX_ATTEMPTS — /health returned $http_code. Retrying in ${HEALTH_CHECK_DELAY_SECONDS}s ..." "warning"
        sleep "$HEALTH_CHECK_DELAY_SECONDS"
        (( attempt++ ))
    done

    log "Timed out waiting for /health to return 200" "error"
    exit 1
}

provision() {
    local interface_name="$1"
    log "Provisioning input for interface: $interface_name"

    local src_dir="${INPUT_DIR}/${interface_name}"
    if [[ ! -d "$src_dir" ]]; then
        log "Interface input directory not found: $src_dir" "error"
        exit 1
    fi

    rm -rf "$STAGING_INPUT_DIR"
    mkdir -p "$STAGING_INPUT_DIR"
    cp -r "${src_dir}/"* "$STAGING_INPUT_DIR/"
    chmod -R o+rX "$STAGING_INPUT_DIR"

    log "Input files staged:"
    find "$STAGING_INPUT_DIR" -type f

    docker run \
        --platform=linux/amd64 \
        --rm \
        --name "$TESTER_NAME" \
        --volume "$DOCKER_VOLUME_TAG":/output \
        --volume "$STAGING_INPUT_DIR":/input:ro \
        alpine:latest \
        sh -c "rm -rf /output/*"
}

invoke() {
    log "Calling POST /invoke (timeout: ${INVOKE_TIMEOUT_SECONDS}s) ..."
    local url="${BASE_URL}/invoke"

    local http_code
    http_code=$(docker run \
        --platform=linux/amd64 \
        --rm \
        --network "$DOCKER_NETWORK_TAG" \
        curlimages/curl:8.10.1 \
        --silent \
        --output /dev/null \
        --write-out "%{http_code}" \
        --max-time "$INVOKE_TIMEOUT_SECONDS" \
        --request POST \
        "$url") || http_code="000"

    show_new_logs

    if [[ "$http_code" == "200" || "$http_code" == "201" ]]; then
        log "/invoke returned $http_code — inference succeeded"
    else
        log "/invoke returned $http_code (expected 200 or 201)" "error"
        exit 1
    fi
}

collect_output() {
    local interface_name="$1"
    log "Collecting output for interface: $interface_name"

    local out_dest="${OUTPUT_DIR}/${interface_name}"
    mkdir -p "$out_dest"

    rm -rf "$STAGING_OUTPUT_DIR"
    mkdir -p "$STAGING_OUTPUT_DIR"

    docker run \
        --platform=linux/amd64 \
        --rm \
        --name "$TESTER_NAME" \
        --volume "$DOCKER_VOLUME_TAG":/output:ro \
        --volume "$STAGING_OUTPUT_DIR":/host_output \
        alpine:latest \
        sh -c "cp -r /output/* /host_output/ 2>/dev/null || true"

    if [[ -d "$STAGING_OUTPUT_DIR" ]] && [[ "$(ls -A "$STAGING_OUTPUT_DIR" 2>/dev/null)" ]]; then
        cp -r "${STAGING_OUTPUT_DIR}/"* "$out_dest/"
        chmod -R o+rw "$out_dest" 2>/dev/null || true
        log "Output files:"
        find "$out_dest" -type f
    else
        log "No files found in /output" "warning"
    fi
}

main "$@"
