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
: "${RAY_TMPDIR:=/tmp/ray}"      # user sets this in init.sh
: "${RAY_STOP_RETRIES:=5}"      # how many times to run ray stop --force
: "${RAY_STOP_INTERVAL:=2}"     # seconds between ray stop attempts

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

ray_start_supports() {
  local flag="$1"
  ray start --help 2>/dev/null | grep -q -- "$flag"
}

# Best-effort: show what is listening on a port
show_port_owner() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lntp 2>/dev/null | grep -E ":${port}\b" || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | grep ":${port} " || true
  else
    echo "(no ss/lsof available to inspect ports)"
  fi
}

is_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | grep -q ":${port} "
  else
    # If we cannot check, assume not in use
    return 1
  fi
}

ray_stop_until_clean() {
  local retries="$RAY_STOP_RETRIES"
  local interval="$RAY_STOP_INTERVAL"

  echo "cleanup: ray stop --force (up to ${retries} attempts)"

  for i in $(seq 1 "${retries}"); do
    echo "cleanup: ray stop attempt ${i}/${retries}"
    # Capture output to help debugging but do not fail the script on non-zero.
    local out
    out=$(ray stop --force 2>&1 || true)
    if [ -n "${out}" ]; then
      echo "cleanup: ray stop output: ${out}"
    fi

    # If ray reports no active processes, we still might have a lingering redis on the port.
    # Prefer checking port availability when possible.
    if ! is_port_in_use "${RAY_PORT}"; then
      echo "cleanup: port ${RAY_PORT} is free after ray stop"
      return 0
    fi

    echo "cleanup: port ${RAY_PORT} still in use; sleeping ${interval}s"
    sleep "${interval}"
  done

  echo "cleanup: ray stop retries exhausted; port ${RAY_PORT} still in use"
  return 1
}

cleanup_ray() {
  if [ "${RAY_CLEANUP}" != "1" ]; then
    echo "cleanup: skipped (RAY_CLEANUP=${RAY_CLEANUP})"
    return 0
  fi

  ray_stop_until_clean || true

  echo "cleanup: kill leftover ray processes"
  pkill -9 -f raylet >/dev/null 2>&1 || true
  pkill -9 -f gcs_server >/dev/null 2>&1 || true
  pkill -9 -f "ray::" >/dev/null 2>&1 || true
  # Try to only kill Ray-managed redis (avoid killing system redis)
  pkill -9 -f "redis-server.*ray" >/dev/null 2>&1 || true

  echo "cleanup: remove local ray state"
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

    # If port is already in use, do NOT start; this is the most common cause of
    # "session name does not match persisted value" (ray connects to existing redis/KV).
    echo "head: preflight check port ${RAY_PORT}"
    if is_port_in_use "${RAY_PORT}"; then
      echo "ERROR: RAY_PORT ${RAY_PORT} is already in use. This usually means an old Ray/redis is still running." >&2
      echo "head: listener(s) on port ${RAY_PORT}:" >&2
      show_port_owner "${RAY_PORT}" >&2
      echo "Please stop the existing process (or pick a different RAY_PORT)." >&2
      exit 1
    fi

    cleanup_ray

    # Re-check after cleanup
    echo "head: re-check port ${RAY_PORT} after cleanup"
    if is_port_in_use "${RAY_PORT}"; then
      echo "ERROR: RAY_PORT ${RAY_PORT} is still in use after cleanup." >&2
      echo "head: listener(s) on port ${RAY_PORT}:" >&2
      show_port_owner "${RAY_PORT}" >&2
      exit 1
    fi

    # Build the most portable ray start command.
    local -a cmd
    cmd=(ray start --head --disable-usage-stats --port="${RAY_PORT}")

    if [ -n "${RAY_NODE_IP}" ] && ray_start_supports "--node-ip-address"; then
      echo "head: using node ip ${RAY_NODE_IP}"
      cmd+=(--node-ip-address="${RAY_NODE_IP}")
    fi

    # Use the user-provided tmpdir if Ray supports it.
    if ray_start_supports "--temp-dir"; then
      cmd+=(--temp-dir="${RAY_TMPDIR}")
    fi

    echo "head: launching: ${cmd[*]}"
    "${cmd[@]}"

    echo "head: started"
    verify_nodes

  else
    echo "worker: connecting to ${MASTER_ADDR}:${RAY_PORT}"
    local -a wcmd
    wcmd=(ray start --address="${MASTER_ADDR}:${RAY_PORT}")

    if [ -n "${RAY_NODE_IP}" ] && ray_start_supports "--node-ip-address"; then
      wcmd+=(--node-ip-address="${RAY_NODE_IP}")
    fi

    if ray_start_supports "--temp-dir"; then
      wcmd+=(--temp-dir="${RAY_TMPDIR}")
    fi

    echo "worker: launching: ${wcmd[*]}"
    "${wcmd[@]}" || true
    ray status >/dev/null 2>&1 || true
  fi
}

init_ray