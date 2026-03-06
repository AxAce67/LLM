#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_URL="${VERSION_URL:-}"
LOCAL_VERSION_FILE="${BASE_DIR}/version.json"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ -z "${VERSION_URL}" ]]; then
  echo "VERSION_URL is required. Example:"
  echo "  VERSION_URL=https://example.com/version.json ./ops/update-app.sh"
  exit 1
fi

echo "[UpdateApp] Fetching remote version file..."
curl -fsSL "${VERSION_URL}" -o "${TMP_DIR}/version.json"

REMOTE_APP_URL="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("app",{}).get("url",""))' "${TMP_DIR}/version.json")"
REMOTE_APP_SHA="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("app",{}).get("sha256",""))' "${TMP_DIR}/version.json")"
REMOTE_APP_VER="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("app_version",""))' "${TMP_DIR}/version.json")"
LOCAL_APP_VER="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("app_version",""))' "${LOCAL_VERSION_FILE}" 2>/dev/null || true)"

if [[ -z "${REMOTE_APP_URL}" ]]; then
  echo "[UpdateApp] app.url is empty. Skip."
  exit 0
fi

if [[ "${REMOTE_APP_VER}" == "${LOCAL_APP_VER}" ]]; then
  echo "[UpdateApp] Already up to date: ${LOCAL_APP_VER}"
  exit 0
fi

echo "[UpdateApp] Downloading app bundle ${REMOTE_APP_VER}..."
curl -fsSL "${REMOTE_APP_URL}" -o "${TMP_DIR}/app.tar.gz"

if [[ -n "${REMOTE_APP_SHA}" ]]; then
  DOWN_SHA="$(shasum -a 256 "${TMP_DIR}/app.tar.gz" | awk '{print $1}')"
  if [[ "${DOWN_SHA}" != "${REMOTE_APP_SHA}" ]]; then
    echo "[UpdateApp] SHA mismatch."
    exit 1
  fi
fi

cp -f "${LOCAL_VERSION_FILE}" "${TMP_DIR}/version.local.backup.json" 2>/dev/null || true
tar -xzf "${TMP_DIR}/app.tar.gz" -C "${BASE_DIR}"
cp -f "${TMP_DIR}/version.json" "${LOCAL_VERSION_FILE}"
echo "[UpdateApp] Updated to app version ${REMOTE_APP_VER}"
