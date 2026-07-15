#!/usr/bin/env bash
# Start the transcript API server on the GPU host.
# Usage: ./deploy/run-server.sh
set -euo pipefail

# Generate an owner-only token file once, then reuse it across restarts.
if [[ -z "${TRANSCRIPT_TOKEN:-}" ]]; then
  TOKEN_FILE="${TRANSCRIPT_TOKEN_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/transcript/server.token}"
  TOKEN_DIR="$(dirname "$TOKEN_FILE")"
  if [[ ! -d "$TOKEN_DIR" ]]; then
    (umask 077; mkdir -p "$TOKEN_DIR")
  fi
  if [[ -s "$TOKEN_FILE" ]]; then
    export TRANSCRIPT_TOKEN="$(<"$TOKEN_FILE")"
  else
    export TRANSCRIPT_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
    (umask 077; printf '%s\n' "$TRANSCRIPT_TOKEN" > "$TOKEN_FILE")
    TOKEN_GENERATED=1
    echo "Saved TRANSCRIPT_TOKEN to $TOKEN_FILE"
  fi
  chmod 600 "$TOKEN_FILE"
fi

if [[ -z "$TRANSCRIPT_TOKEN" ]]; then
  echo "TRANSCRIPT_TOKEN is empty" >&2
  exit 1
fi

if [[ "${TOKEN_GENERATED:-}" ]]; then
  echo "Generated TRANSCRIPT_TOKEN=${TRANSCRIPT_TOKEN}"
  echo "Set this on your Mac:  export TRANSCRIPT_TOKEN=${TRANSCRIPT_TOKEN}"
fi

HOST="${TRANSCRIPT_HOST:-0.0.0.0}"
PORT="${TRANSCRIPT_PORT:-8000}"
MODEL="${TRANSCRIPT_MODEL:-large-v3}"

exec transcript-server --host "$HOST" --port "$PORT" --model "$MODEL"
