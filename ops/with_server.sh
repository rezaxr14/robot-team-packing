#!/usr/bin/env bash
# Start a vLLM server, wait until it answers, run a command, always shut the server down.
# usage: ops/with_server.sh <name> <model> <port> <util> -- <command...>
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME="$1"; MODEL="$2"; PORT="$3"; UTIL="$4"; shift 4
[ "$1" = "--" ] && shift
LOG="$RTP_ROOT/reports/logs/server_${NAME}.log"; mkdir -p "$(dirname "$LOG")"
"$RTP_ROOT/ops/serve.sh" "$NAME" "$MODEL" "$PORT" "$UTIL" > "$LOG" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true' EXIT
for i in $(seq 1 ${WAIT_TRIES:-180}); do
  if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  kill -0 $SPID 2>/dev/null || { echo "server died; last log lines:"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
[ "${READY:-0}" = 1 ] || { echo "server not ready; last log lines:"; tail -30 "$LOG"; exit 1; }
echo "server $NAME ready on :$PORT"
"$@"
