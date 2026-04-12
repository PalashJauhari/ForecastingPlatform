#!/usr/bin/env bash
# Launch Forecasting Platform: API (uvicorn) + UI (Plotly Dash)

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f .env ]; then
  echo "Warning: .env not found. Create it and add OPENAI_API_KEY=your-key"
fi

# Start API in background. No --reload so the process inherits PYTHONPATH (reloader subprocess would not).
echo "Starting API (uvicorn) on http://127.0.0.1:8000 ..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Stop API when script exits (e.g. Ctrl+C)
trap 'kill $API_PID 2>/dev/null' EXIT

# Give API a moment to bind
sleep 2

# Start UI (foreground; Ctrl+C will also trigger trap and kill API)
echo "Starting UI (Dash) on http://127.0.0.1:8501 ..."
python ui/dash_app.py
