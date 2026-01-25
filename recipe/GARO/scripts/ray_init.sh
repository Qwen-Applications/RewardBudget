#!/bin/bash
set -euo pipefail

# -------------------------------
# Required env (must be provided)
# -------------------------------
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "ERROR: MASTER_ADDR is not set" >&2
  exit 1
fi

if [ -z "${RAY_PORT:-}" ]; then
  echo "ERROR: RAY_PORT is not set" >&2
  exit 1
fi

if ! [[ "${RAY_PORT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: RAY_PORT must be numeric, got '${RAY_PORT}'" >&2
  exit 1
fi

# -------------------------------
# Optional env
# -------------------------------
: "${NNODES:=1}"
: "${RANK:=0}"
: "${RAY_CLEANUP:=1}"
: "${RAY_NODE_IP:=}"            # optional

# Caller said they already set this in init.sh. If not set, default to /tmp/ray.
: "${RAY_TMPDIR:=/tmp/ray}"

# Safety: never delete anything under /sls-log
is_under_sls_log() {
  local p="$1"
  case "$p" in
    /sls-log|/sls-log/*) return 0 ;;
  esac
  if command -v realpath >/dev/null 2>&1; then
    local rp
    rp=$(realpath -m "$p" 2>/dev/null || true)
    case "$rp" in
      /sls-log|/sls-log/*) return 0 ;;
    esac
  fi
  return 1
}

cleanup_ray() {
  if [ "${RAY_CLEANUP}" != "1" ]; then
    echo "cleanup: skipped (RAY_CLEANUP=${RAY_CLEANUP})"
    return 0
  fi

  echo "cleanup: ray stop --force"
  ray stop --force || true

  echo "cleanup: kill leftover processes"
  pkill -9 -f raylet >/dev/null 2>&1 || true
  pkill -9 -f gcs_server >/dev/null 2>&1 || true
  pkill -9 -f "ray::" >/dev/null 2>&1 || true

  echo "cleanup: remove local ray state"
  # These locations are where Ray most commonly persists session state.
  # Removing them prevents 'session name does not match persisted value' errors.
  for p in /tmp/ray /tmp/ray_session* /dev/shm/ray "${HOME}/.ray" "${RAY_TMPDIR}"; do
    if is_under_sls_log "$p"; then
      echo "cleanup: SKIP deleting $p (under /sls-log)"
      continue
    fi
    if [ -e "$p" ] || [ -L "$p" ]; then
      echo "cleanup: rm -rf $p"
      rm -rf "$p" 2>/dev/null || true
    fi
  done

  echo "cleanup: done"
}

verify_nodes() {
  local expected="$NNODES"
  local retries=30
  local interval=2

  echo "verify: expecting ${expected} node(s)"
  for i in $(seq 1 "$retries"); do
    # Count ALIVE mentions (works across many Ray versions)
    local alive
    alive=$(ray status 2>/dev/null | grep -E "\bALIVE\b" | wc -l | tr -d ' ' || echo 0)
    if [ "${alive}" -ge "${expected}" ]; then
      echo "verify: OK (${alive}/${expected})"
      return 0
    fi
    echo "verify: waiting... (${alive}/${expected}) attempt ${i}/${retries}"
    sleep "$interval"
  done

  echo "verify: FAILED"
  ray status 2>/dev/null || true
  return 1
}

init_ray() {
  echo "[Ray ENV] MASTER_ADDR=${MASTER_ADDR} RAY_PORT=${RAY_PORT} NNODES=${NNODES} RANK=${RANK} RAY_TMPDIR=${RAY_TMPDIR}"

  if [ "${RANK}" -eq 0 ]; then
    echo "head: starting on ${MASTER_ADDR}:${RAY_PORT}"
    cleanup_ray

    # Most portable flags only
    if [ -n "${RAY_NODE_IP}" ]; then
      echo "head: using node ip ${RAY_NODE_IP}"
      ray start --head --port="${RAY_PORT}" --node-ip-address="${RAY_NODE_IP}"
    else
      ray start --head --port="${RAY_PORT}"
    fi

    echo "head: started"
    verify_nodes
  else
    echo "worker: connecting to ${MASTER_ADDR}:${RAY_PORT}"
    if [ -n "${RAY_NODE_IP}" ]; then
      ray start --address="${MASTER_ADDR}:${RAY_PORT}" --node-ip-address="${RAY_NODE_IP}" || true
    else
      ray start --address="${MASTER_ADDR}:${RAY_PORT}" || true
    fi

    # Basic check
    ray status >/dev/null 2>&1 || true
  fi
}

init_ray