#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ADMIN_TOKEN="${ADMIN_API_TOKEN:-}"
RUN_CONTROL_TEST="${RUN_CONTROL_TEST:-0}"

ok() { echo "[OK] $*"; }
ng() { echo "[NG] $*" >&2; exit 1; }

check_get() {
  local path="$1"
  local must="$2"
  local tmp
  tmp="$(mktemp)"
  local code
  code="$(curl -sS -o "$tmp" -w "%{http_code}" "${BASE_URL}${path}" || true)"
  if [[ "$code" != "200" ]]; then
    cat "$tmp" >&2 || true
    rm -f "$tmp"
    ng "GET ${path} failed (HTTP ${code})"
  fi
  if ! grep -q "$must" "$tmp"; then
    cat "$tmp" >&2 || true
    rm -f "$tmp"
    ng "GET ${path} unexpected body (missing: ${must})"
  fi
  rm -f "$tmp"
  ok "GET ${path}"
}

check_get "/api/runtime-config" "system_role"
check_get "/api/status" "is_running"
check_get "/api/nodes" "\"status\":\"success\""
check_get "/api/policies" "\"status\":\"success\""
check_get "/api/evals" "\"status\":\"success\""
check_get "/api/models" "\"status\":\"success\""
check_get "/api/datasets" "\"status\":\"success\""
check_get "/api/evals/status" "status"
check_get "/api/migration/hf-train/status" "status"
check_get "/api/migration/gguf-export/status" "status"

if [[ "$RUN_CONTROL_TEST" == "1" ]]; then
  [[ -n "$ADMIN_TOKEN" ]] || ng "RUN_CONTROL_TEST=1 requires ADMIN_API_TOKEN"

  code="$(curl -sS -o /tmp/hc_control_start.json -w "%{http_code}" \
    -X POST "${BASE_URL}/api/control" \
    -H "Content-Type: application/json" \
    -H "x-admin-token: ${ADMIN_TOKEN}" \
    -d '{"action":"start"}' || true)"
  [[ "$code" == "200" ]] || ng "POST /api/control start failed (HTTP ${code})"
  ok "POST /api/control start"

  code="$(curl -sS -o /tmp/hc_control_stop.json -w "%{http_code}" \
    -X POST "${BASE_URL}/api/control" \
    -H "Content-Type: application/json" \
    -H "x-admin-token: ${ADMIN_TOKEN}" \
    -d '{"action":"stop"}' || true)"
  [[ "$code" == "200" ]] || ng "POST /api/control stop failed (HTTP ${code})"
  ok "POST /api/control stop"
fi

echo "healthcheck passed"
