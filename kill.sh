#!/usr/bin/env bash
# Stop Forecasting Platform API (uvicorn) and UI (Dash) by freeing their listen ports.
#
# Matches ports used by start.sh:
#   API  http://127.0.0.1:8000
#   UI   http://127.0.0.1:8501

set -euo pipefail

API_PORT=8000
UI_PORT=8501

free_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    echo "$label: port $port is free."
    return 0
  fi
  echo "$label: port $port in use by PID(s): $pids — terminating."
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "$label: port $port still busy — SIGKILL PID(s): $pids."
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "$label: warning — port $port may still be in use."
    return 1
  fi
  echo "$label: port $port is free."
}

free_port "$API_PORT" "API (backend)"
free_port "$UI_PORT" "UI (frontend)"
