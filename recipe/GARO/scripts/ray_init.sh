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
: "${RAY_CLEANUP:=1}"
: "${RAY_START_TIMEOUT:=45}"   # seconds; fail fast if `ray start` hangs
: "${RAY_TMPDIR:=/tmp/ray_${USER:-user}_${MASTER_ADDR}_${RAY_PORT}_$$_$(date +%s)}"
: "${RAY_NODE_IP:=}"           # if empty, auto-detect and pass to ray start
: "${RAY_DEBUG:=0}"            # set to 1 to enable extra shell tracing

if [ "${RAY_DEBUG}" = "1" ]; then
    echo "ray_init.sh: RAY_DEBUG=1 enabling xtrace"
    set -x
fi

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

on_term() {
    local sig="$1"
    echo "ray_init.sh received signal: ${sig}"
    echo "--- ray status (on signal) ---"
    ray status 2>/dev/null || true
    echo "--- ray nodes (on signal) ---"
    ray list nodes 2>/dev/null || true
    echo "--- processes (on signal) ---"
    ps aux | egrep "raylet|gcs_server|redis|dashboard|ray start|timeout" | grep -v grep || true
    exit 143
}
trap 'on_term SIGTERM' SIGTERM
trap 'on_term SIGINT' SIGINT

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

detect_node_ip() {
    # 1) If user explicitly provided RAY_NODE_IP, trust it
    if [ -n "${RAY_NODE_IP}" ]; then
        echo "${RAY_NODE_IP}"
        return 0
    fi

    # 2) If MASTER_ADDR is already an IP, prefer it
    if [[ "${MASTER_ADDR}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "${MASTER_ADDR}"
        return 0
    fi

    # 3) Try python-based resolution + local IP inference (works in minimal images)
    if command -v python >/dev/null 2>&1; then
        python - <<'PY'
import socket
import os
master = os.environ.get('MASTER_ADDR','')
# Resolve MASTER_ADDR to an IP (may fail)
resolved = ''
try:
    resolved = socket.gethostbyname(master) if master else ''
except Exception:
    resolved = ''
# Infer local source IP by UDP connect trick
local_ip = ''
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('1.1.1.1', 80))
    local_ip = s.getsockname()[0]
except Exception:
    local_ip = ''
finally:
    s.close()
# Prefer resolved if it matches inferred local subnet (first two octets) else use local_ip
if resolved and local_ip and resolved.split('.')[:2] == local_ip.split('.')[:2]:
    print(resolved)
elif local_ip:
    print(local_ip)
elif resolved:
    print(resolved)
else:
    print('')
PY
        return 0
    fi

    # 4) Last resort: let Ray decide
    echo ""
    return 0
}

run_ray_start_with_timeout() {
    # Usage: run_ray_start_with_timeout <command...>
    # Runs ray start with a timeout to prevent hanging forever.
    local timeout_s="${RAY_START_TIMEOUT}"

    if command -v timeout >/dev/null 2>&1; then
        echo "run_ray_start_with_timeout: using external timeout (${timeout_s}s)"
        echo "run_ray_start_with_timeout: timeout path=$(command -v timeout)"
        timeout --version >/dev/null 2>&1 && timeout --version | head -n 1 || true
        echo "run_ray_start_with_timeout: launching command: $*"

        # Prefer --foreground if supported (GNU timeout) to keep signals/output behavior predictable.
        if timeout --help 2>/dev/null | grep -q -- '--foreground'; then
            timeout --foreground "${timeout_s}s" "$@"
        else
            timeout "${timeout_s}s" "$@"
        fi
        local rc=$?
        echo "run_ray_start_with_timeout: external timeout finished (rc=${rc})"
        return ${rc}
    fi

    echo "run_ray_start_with_timeout: using watchdog (${timeout_s}s)"
    # Fallback: background + watchdog (no external timeout command)
    "$@" &
    local pid=$!

    local waited=0
    while kill -0 "${pid}" >/dev/null 2>&1; do
        if [ "${waited}" -ge "${timeout_s}" ]; then
            echo "ray start timeout after ${timeout_s}s; killing pid ${pid}"
            kill -9 "${pid}" >/dev/null 2>&1 || true
            return 124
        fi
        echo "ray start still running... ${waited}s"
        sleep 1
        waited=$((waited + 1))
    done

    wait "${pid}"
    return $?
}

cleanup_ray() {
    # Best-effort cleanup of stale Ray state (only when enabled)
    if [ "${RAY_CLEANUP}" != "1" ]; then
        echo "RAY_CLEANUP=${RAY_CLEANUP}; skipping cleanup"
        return 0
    fi

    echo "Cleaning up stale Ray state..."
    echo "cleanup: before ray stop"
    ray stop --force >/dev/null 2>&1 || true
    echo "cleanup: after ray stop"
    echo "cleanup: killing ray processes"
    pkill -9 -f raylet >/dev/null 2>&1 || true
    pkill -9 -f gcs_server >/dev/null 2>&1 || true
    pkill -9 -f "ray::" >/dev/null 2>&1 || true

    echo "cleanup: removing ray temp directories"
    rm -rf /tmp/ray /tmp/ray_session* /dev/shm/ray "${HOME}/.ray" 2>/dev/null || true

    # Clean our run tempdir if it already exists
    rm -rf "${RAY_TMPDIR}" 2>/dev/null || true

    echo "cleanup: done"
}

wait_for_ray_ready() {
    local retries=${1:-30}
    local interval=${2:-2}

    for _ in $(seq 1 "$retries"); do
        echo "wait_for_ray_ready: probing ray status"
        if ray status >/dev/null 2>&1; then
            echo "wait_for_ray_ready: ray status OK"
            return 0
        fi
        sleep "$interval"
    done
    echo "wait_for_ray_ready: timed out waiting for ray"
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
        echo "head: checking port availability"

        local node_ip
        node_ip=$(detect_node_ip)
        if [ -n "${node_ip}" ]; then
            echo "head: using node ip address ${node_ip}"
            if [[ "${MASTER_ADDR}" =~ ^[0-9] ]]; then
                if [[ "${MASTER_ADDR}" != "${node_ip}" ]]; then
                    echo "WARNING: MASTER_ADDR (${MASTER_ADDR}) != node_ip (${node_ip})"
                fi
            fi
        else
            echo "head: node ip auto-detect failed; letting Ray decide"
        fi

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

        echo "head: invoking cleanup_ray"
        cleanup_ray
        echo "head: cleanup_ray finished"
        mkdir -p "${RAY_TMPDIR}" || true

        echo "head: running ray start --head"
        echo "head: ray start args: port=${RAY_PORT} gcs=${RAY_GCS_PORT} dashboard=${RAY_DASHBOARD_PORT} tmp=${RAY_TMPDIR} node_ip=${node_ip}" 

        # Start head with an isolated temp dir to avoid session-name collisions.
        # Wrap with timeout because `ray start` can hang in some environments.
        set +e
        if [ -n "${node_ip}" ]; then
            run_ray_start_with_timeout ray start --head \
                --disable-usage-stats \
                --node-ip-address="${node_ip}" \
                --port="${RAY_PORT}" \
                --gcs-server-port="${RAY_GCS_PORT}" \
                --dashboard-host=0.0.0.0 \
                --dashboard-port="${RAY_DASHBOARD_PORT}" \
                --temp-dir="${RAY_TMPDIR}"
        else
            run_ray_start_with_timeout ray start --head \
                --disable-usage-stats \
                --port="${RAY_PORT}" \
                --gcs-server-port="${RAY_GCS_PORT}" \
                --dashboard-host=0.0.0.0 \
                --dashboard-port="${RAY_DASHBOARD_PORT}" \
                --temp-dir="${RAY_TMPDIR}"
        fi
        local ray_rc=$?
        set -e
        echo "head: ray start command returned (rc=${ray_rc})"
        if [ "${ray_rc}" -ne 0 ]; then
            echo "head: ray start failed or timed out; dumping quick diagnostics"
            echo "head: ports listening snapshot"
            if command -v ss >/dev/null 2>&1; then
                ss -lntp 2>/dev/null || true
            elif command -v netstat >/dev/null 2>&1; then
                netstat -lntp 2>/dev/null || true
            fi
            die "Ray head start failed (rc=${ray_rc})."
        fi

        echo "head: waiting for ray to become ready"
        if ! wait_for_ray_ready 30 2; then
            die "Ray head did not become ready in time (MASTER_ADDR=${MASTER_ADDR}, RAY_PORT=${RAY_PORT})."
        fi

        echo "Ray head node is ready (dashboard_port=${RAY_DASHBOARD_PORT}, gcs_port=${RAY_GCS_PORT}, tmp=${RAY_TMPDIR})"

    else
        echo "Connecting to Ray head node at ${MASTER_ADDR}:${RAY_PORT} ..."

        # Small deterministic stagger to reduce connection storms
        sleep $(( (RANK % 5) + 1 ))

        for i in $(seq 1 "${max_retries}"); do
            echo "worker: running ray start --address"
            local worker_ip
            worker_ip=$(detect_node_ip)
            set +e
            if [ -n "${worker_ip}" ]; then
                run_ray_start_with_timeout ray start --address="${MASTER_ADDR}:${RAY_PORT}" --node-ip-address="${worker_ip}" --temp-dir="${RAY_TMPDIR}"
            else
                run_ray_start_with_timeout ray start --address="${MASTER_ADDR}:${RAY_PORT}" --temp-dir="${RAY_TMPDIR}"
            fi
            set -e
            echo "worker: ray start command returned"

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
