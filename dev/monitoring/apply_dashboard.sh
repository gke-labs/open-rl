#!/usr/bin/env bash
#
# Idempotent create-or-update for the Open-RL Cloud Monitoring dashboard.
#
# Usage:
#   ./scripts/apply_dashboard.sh <PROJECT_ID>
#   PROJECT_ID=my-project ./scripts/apply_dashboard.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_FILE="$SCRIPT_DIR/openrl-performance.dashboard.json"
DISPLAY_NAME="Open-RL: Accelerator Performance"

PROJECT_ID="${1:-${PROJECT_ID:-}}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Usage: $0 <PROJECT_ID>"
  echo "  or:  PROJECT_ID=<project> $0"
  exit 1
fi

if [[ ! -f "$DASHBOARD_FILE" ]]; then
  echo "Error: Dashboard file not found: $DASHBOARD_FILE"
  exit 1
fi

echo "Looking for existing dashboard: \"$DISPLAY_NAME\" in project $PROJECT_ID ..."

EXISTING=$(gcloud monitoring dashboards list \
  --project="$PROJECT_ID" \
  --filter="displayName=\"$DISPLAY_NAME\"" \
  --format="value(name)" 2>/dev/null || true)

if [[ -n "$EXISTING" ]]; then
  DASHBOARD_ID=$(basename "$EXISTING")
  echo "Found existing dashboard: $DASHBOARD_ID"
  echo "Updating..."
  # Google Cloud Monitoring requires updates to provide the existing etag
  export ETAG=$(gcloud monitoring dashboards describe "$DASHBOARD_ID" --project="$PROJECT_ID" --format="value(etag)")
  
  # Create a temporary file with the etag injected
  TMP_DASHBOARD=$(mktemp)
  jq --arg etag "$ETAG" '.etag = $etag' "$DASHBOARD_FILE" > "$TMP_DASHBOARD"

  gcloud monitoring dashboards update "$DASHBOARD_ID" \
    --config-from-file="$TMP_DASHBOARD" \
    --project="$PROJECT_ID"
    
  rm -f "$TMP_DASHBOARD"
else
  echo "No existing dashboard found. Creating..."
  gcloud monitoring dashboards create \
    --config-from-file="$DASHBOARD_FILE" \
    --project="$PROJECT_ID"
fi

echo ""
echo "Done. View at:"
echo "  https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
