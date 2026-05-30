#!/usr/bin/env bash
# Start the transcript API server on the GPU host.
# Usage: ./deploy/run-server.sh
set -euo pipefail

# Generate a token on first run if none is set, and print it so the Mac can use it.
if [[ -z "${TRANSCRIPT_TOKEN:-}" ]]; then
  export TRANSCRIPT_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
  echo "Generated TRANSCRIPT_TOKEN=${TRANSCRIPT_TOKEN}"
  echo "Set this on your Mac:  export TRANSCRIPT_TOKEN=${TRANSCRIPT_TOKEN}"
fi

HOST="${TRANSCRIPT_HOST:-0.0.0.0}"
PORT="${TRANSCRIPT_PORT:-8000}"
MODEL="${TRANSCRIPT_MODEL:-large-v3}"

exec transcript-server --host "$HOST" --port "$PORT" --model "$MODEL"
