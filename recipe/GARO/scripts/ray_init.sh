#!/bin/bash

set -euo pipefail

# -------------------------------
# Required env (must be provided)
# -------------------------------
if [ -z "${MASTER_ADDR:-}" ]; then
    echo "ERROR: MASTER_ADDR is not set. Please export MASTER_ADDR before running this script." >&2
    exit 1
fi

if [ -z "${RAY_PORT:-}" ]; then
    echo "ERROR: RAY_PORT is not set. Please export RAY_PORT before running this script." >&2
    exit 1
fi

# Basic sanity check for RAY_PORT (numeric)
if ! [[ "${RAY_PORT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RAY_PORT must be a numeric port, got '${RAY_PORT}'." >&2
    exit 1
fi

# -------------------------------
# Optional env (safe defaults)
# -------------------------------
: "${NNODES:=1}"
: "${RANK:=0}"
: "${RAY_DASHBOARD_PORT:=8265}"
: "${RAY_GCS_PORT:=10001}"

# Per-run Ray temp dir to avoid session collisions; can be overridden.
# (Does NOT change MASTER_ADDR / RAY_PORT contract.)
: "${RAY_TMPDIR:=/tmp/ray_${USER:-user}_${MASTER_ADDR}_${RAY_PORT}_$$_$(date +%s)}"

# Whether to aggressively clean up stale ray processes/dirs before starting head.
: "${RAY_CLEANUP:=1}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# Print useful diagnostics on failure
on_error() {
    local rc=$?
    echo "ray_init.sh failed with exit code ${rc}"
    echo "ENV: MASTER_ADDR=${MASTER_ADDR} RAY_PORT=${RAY_PORT} NNODES=${NNODES} RANK=${RANK} RAY_TMPDIR=${RAY_TMPDIR}"

    # Best-effort diagnostics
    echo "--- ray status ---"
    ray status 2>/dev/null || true

    echo "--- ray nodes (if supported) ---"
    ray list nodes 2>/dev/null || true

    echo "--- processes ---"
    ps aux | egrep "raylet|gcs_server|redis|dashboard" | grep -v grep || true

    echo "--- temp dir listing ---"
    ls -la "${RAY_TMPDIR}" 2>/dev/null || true

    exit ${rc}
}
trap on_error ERR

is_port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | grep -q ":${port} "
    else
        # If we cannot check, do not block execution
        return 1
    fi
}

cleanup_ray() {
    # Best-effort cleanup of stale Ray state (only when enabled)
    if [ "${RAY_CLEANUP}" != "1" ]; then
        echo "RAY_CLEANUP=${RAY_CLEANUP}; skipping cleanup"
        return 0
    fi

    echo "Cleaning up stale Ray state..."
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f raylet >/dev/null 2>&1 || true
    pkill -9 -f gcs_server >/dev/null 2>&1 || true
    pkill -9 -f "ray::" >/dev/null 2>&1 || true

    rm -rf /tmp/ray /tmp/ray_session* /dev/shm/ray "${HOME}/.ray" 2>/dev/null || true

    # Clean our run tempdir if it already exists
    rm -rf "${RAY_TMPDIR}" 2>/dev/null || true
}

wait_for_ray_ready() {
    local retries=${1:-30}
    local interval=${2:-2}

    for _ in $(seq 1 "$retries"); do
        if ray status >/dev/null 2>&1; then
            return 0
        fi
        sleep "$interval"
    done
    return 1
}

get_alive_node_count() {
    # Prefer `ray list nodes` (stable across versions) and fallback to `ray status`
    if ray list nodes >/dev/null 2>&1; then
        ray list nodes 2>/dev/null | grep -E "\bALIVE\b" | wc -l | tr -d ' '
        return 0
    fi

    # Fallback: try to count ALIVE mentions in `ray status`
    ray status 2>/dev/null | grep -E "\bALIVE\b" | wc -l | tr -d ' '
}

verify_ray_nodes() {
    local expected_nodes="$NNODES"
    local max_retries=30
    local retry_interval=10

    echo "Verifying Ray cluster node count (expecting ${expected_nodes} nodes)..."

    for i in $(seq 1 "$max_retries"); do
        local node_count
        node_count=$(get_alive_node_count || echo 0)

        if [ "${node_count}" -eq "${expected_nodes}" ]; then
            echo "Ray cluster node verification successful: ${node_count}/${expected_nodes} nodes"
            return 0
        fi

        if [ "${i}" -eq "${max_retries}" ]; then
            echo "Node verification failed: found ${node_count}/${expected_nodes} nodes"
            return 1
        fi

        echo "Waiting for all nodes to be registered... (${node_count}/${expected_nodes}) attempt ${i}/${max_retries}"
        sleep "${retry_interval}"
    done

    return 1
}

init_ray() {
    local max_retries=30
    local retry_interval=10

    # Strict contract: do not mutate MASTER_ADDR / RAY_PORT
    echo "[Ray ENV] MASTER_ADDR=${MASTER_ADDR} RAY_PORT=${RAY_PORT} NNODES=${NNODES} RANK=${RANK}"

    if [ "${RANK}" -eq 0 ]; then
        echo "Starting Ray head node at ${MASTER_ADDR}:${RAY_PORT} ..."

        # Guard: if the provided ports are already in use, fail fast and instruct cleanup.
        if is_port_in_use "${RAY_PORT}"; then
            die "RAY_PORT ${RAY_PORT} is already in use. Please stop the existing Ray/processes (e.g., 'ray stop --force') or pick another RAY_PORT."
        fi
        if is_port_in_use "${RAY_DASHBOARD_PORT}"; then
            die "RAY_DASHBOARD_PORT ${RAY_DASHBOARD_PORT} is already in use. Please stop the existing process or set a different RAY_DASHBOARD_PORT."
        fi
        if is_port_in_use "${RAY_GCS_PORT}"; then
            die "RAY_GCS_PORT ${RAY_GCS_PORT} is already in use. Please stop the existing process or set a different RAY_GCS_PORT."
        fi

        cleanup_ray
        mkdir -p "${RAY_TMPDIR}" || true

        # Start head with an isolated temp dir to avoid session-name collisions.
        ray start --head \
            --port="${RAY_PORT}" \
            --gcs-server-port="${RAY_GCS_PORT}" \
            --dashboard-host=0.0.0.0 \
            --dashboard-port="${RAY_DASHBOARD_PORT}" \
            --temp-dir="${RAY_TMPDIR}"

        if ! wait_for_ray_ready 30 2; then
            die "Ray head did not become ready in time (MASTER_ADDR=${MASTER_ADDR}, RAY_PORT=${RAY_PORT})."
        fi

        echo "Ray head node is ready (dashboard_port=${RAY_DASHBOARD_PORT}, gcs_port=${RAY_GCS_PORT}, tmp=${RAY_TMPDIR})"

    else
        echo "Connecting to Ray head node at ${MASTER_ADDR}:${RAY_PORT} ..."

        # Small deterministic stagger to reduce connection storms
        sleep $(( (RANK % 5) + 1 ))

        for i in $(seq 1 "${max_retries}"); do
            # Use the provided address strictly
            ray start --address="${MASTER_ADDR}:${RAY_PORT}" --temp-dir="${RAY_TMPDIR}" || true

            # Give raylet a moment to come up
            sleep 3

            if ray status >/dev/null 2>&1; then
                echo "Successfully connected to Ray head node"
                break
            fi

            if [ "${i}" -eq "${max_retries}" ]; then
                die "Failed to connect to Ray head node after ${max_retries} attempts (MASTER_ADDR=${MASTER_ADDR}, RAY_PORT=${RAY_PORT})."
            fi

            echo "Retrying connection to Ray head node... (${i}/${max_retries})"
            sleep "${retry_interval}"
        done
    fi

    # Verify expected nodes (works for single-node too)
    if ! verify_ray_nodes; then
        die "Ray cluster node verification failed."
    fi
}
