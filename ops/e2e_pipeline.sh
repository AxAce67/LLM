#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_API_TOKEN="${ADMIN_API_TOKEN:-}"

echo "[E2E] Base URL: ${BASE_URL}"
HDR=()
if [[ -n "${ADMIN_API_TOKEN}" ]]; then
  HDR=(-H "x-admin-token: ${ADMIN_API_TOKEN}")
fi

status_json="$(curl -fsSL "${BASE_URL}/api/status")"
echo "${status_json}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "stats" in d and "system" in d'
echo "[E2E] status ok"

models_json="$(curl -fsSL "${BASE_URL}/api/models")"
echo "${models_json}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("status")=="success"'
echo "[E2E] models ok"

datasets_json="$(curl -fsSL "${BASE_URL}/api/datasets")"
echo "${datasets_json}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("status")=="success"'
echo "[E2E] datasets ok"

policies_json="$(curl -fsSL "${BASE_URL}/api/policies")"
echo "${policies_json}" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("status")=="success"'
echo "[E2E] policies ok"

curl -fsSL -X POST "${BASE_URL}/api/evals/run" "${HDR[@]}" >/dev/null || true
echo "[E2E] eval trigger sent"

echo "[E2E] done"
