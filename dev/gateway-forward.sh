#!/usr/bin/env bash
# Keep a port-forward to the gateway alive.
#
# kubectl port-forward can wedge without exiting: it logs
# "error creating error stream ... Timeout occurred" and every request hangs,
# so a loop that only restarts on exit never helps. This one probes the
# gateway's health endpoint through the forward and restarts on failure.
#
#   dev/gateway-forward.sh [namespace] [local-port]
set -u
NAMESPACE="${1:-openrl-system}"
PORT="${2:-18000}"
LOG="${GATEWAY_FORWARD_LOG:-/tmp/gateway-forward.log}"

while true; do
  kubectl -n "$NAMESPACE" port-forward svc/open-rl-gateway-service "$PORT:8000" >>"$LOG" 2>&1 &
  pf=$!
  sleep 3
  while kill -0 "$pf" 2>/dev/null; do
    if ! curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/api/v1/healthz"; then
      echo "$(date -u +%FT%TZ) health probe failed; restarting port-forward" >>"$LOG"
      kill "$pf" 2>/dev/null
      sleep 1
      kill -9 "$pf" 2>/dev/null
      break
    fi
    sleep 10
  done
  wait "$pf" 2>/dev/null
  sleep 2
done
