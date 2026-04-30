#!/usr/bin/env bash
# Launch Forecasting Platform: API (uvicorn) + UI (Plotly Dash).
#
# Hardened launcher:
# - Frees stale processes on the API/UI ports before starting (prevents the
#   "errno 48: address already in use" silent half-launch where Dash starts
#   but the new uvicorn dies, leaving Dash to talk to an old backend).
# - Aborts loudly if the API fails to bind, instead of silently starting Dash
#   against a dead backend.
# - Cleans up both API and UI on Ctrl+C / terminal close / kill, with a
#   SIGKILL fallback if the graceful SIGTERM is ignored.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f .env ]; then
  echo "Warning: .env not found. Create it and add OPENAI_API_KEY=your-key"
fi

API_PORT=8000
UI_PORT=8501
API_PID=""
UI_PID=""

# Kill any process currently bound to ``port`` so a fresh launch can start.
free_port() {
  local port="$1"
  local pids
  pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Port $port is in use by PID(s): $pids — terminating."
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [ -n "$pids" ]; then
      echo "Port $port still busy — sending SIGKILL to PID(s): $pids."
      kill -9 $pids 2>/dev/null || true
      sleep 1
    fi
  fi
}

# Stop child processes (API + UI) and free their ports on shell exit.
cleanup() {
  trap - EXIT INT TERM HUP
  local pids=""
  [ -n "$API_PID" ] && pids="$pids $API_PID"
  [ -n "$UI_PID" ] && pids="$pids $UI_PID"

  for pid in $pids; do
    kill "$pid" 2>/dev/null || true
  done

  for _ in 1 2 3; do
    sleep 1
    local alive=""
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        alive="$alive $pid"
      fi
    done
    [ -z "$alive" ] && break
  done

  for pid in $pids; do
    kill -9 "$pid" 2>/dev/null || true
  done

  free_port "$API_PORT"
  free_port "$UI_PORT"
}
trap cleanup EXIT INT TERM HUP

free_port "$API_PORT"
free_port "$UI_PORT"

# Start API in background. No --reload so the process inherits PYTHONPATH (reloader subprocess would not).
echo "Starting API (uvicorn) on http://127.0.0.1:$API_PORT ..."
uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

# Verify the API actually bound on the port; abort the launcher if it died early.
bound=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  if lsof -nP -iTCP:"$API_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    bound=1
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API process exited before binding port $API_PORT. Check the traceback above."
    exit 1
  fi
done
if [ "$bound" != "1" ]; then
  echo "API did not bind on port $API_PORT within 10s. Aborting."
  exit 1
fi

# Start UI in background and wait on it. Closing the UI (Ctrl+C) triggers cleanup
# via the trap, which also tears down the API.
echo "Starting UI (Dash) on http://127.0.0.1:$UI_PORT ..."
python ui/dash_app.py &
UI_PID=$!
wait "$UI_PID"
