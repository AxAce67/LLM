#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_URL="${VERSION_URL:-}"
LOCAL_VERSION_FILE="${BASE_DIR}/version.json"
TARGET_MODEL_PATH="${BASE_DIR}/checkpoints/ckpt_production.pt"
BACKUP_MODEL_PATH="${BASE_DIR}/checkpoints/ckpt_production.prev.pt"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ -z "${VERSION_URL}" ]]; then
  echo "VERSION_URL is required. Example:"
  echo "  VERSION_URL=https://example.com/version.json ./ops/update-model.sh"
  exit 1
fi

echo "[UpdateModel] Fetching remote version file..."
curl -fsSL "${VERSION_URL}" -o "${TMP_DIR}/version.json"

REMOTE_MODEL_URL="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("model",{}).get("url",""))' "${TMP_DIR}/version.json")"
REMOTE_MODEL_SHA="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("model",{}).get("sha256",""))' "${TMP_DIR}/version.json")"
REMOTE_MODEL_VER="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("model_version",""))' "${TMP_DIR}/version.json")"
LOCAL_MODEL_VER="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("model_version",""))' "${LOCAL_VERSION_FILE}" 2>/dev/null || true)"

if [[ -z "${REMOTE_MODEL_URL}" ]]; then
  echo "[UpdateModel] model.url is empty. Skip."
  exit 0
fi

if [[ "${REMOTE_MODEL_VER}" == "${LOCAL_MODEL_VER}" ]]; then
  echo "[UpdateModel] Already up to date: ${LOCAL_MODEL_VER}"
  exit 0
fi

mkdir -p "${BASE_DIR}/checkpoints"
echo "[UpdateModel] Downloading model ${REMOTE_MODEL_VER}..."
curl -fsSL "${REMOTE_MODEL_URL}" -o "${TMP_DIR}/model.bin"

if [[ -n "${REMOTE_MODEL_SHA}" ]]; then
  DOWN_SHA="$(shasum -a 256 "${TMP_DIR}/model.bin" | awk '{print $1}')"
  if [[ "${DOWN_SHA}" != "${REMOTE_MODEL_SHA}" ]]; then
    echo "[UpdateModel] SHA mismatch."
    exit 1
  fi
fi

if [[ -f "${TARGET_MODEL_PATH}" ]]; then
  cp -f "${TARGET_MODEL_PATH}" "${BACKUP_MODEL_PATH}"
fi

cp -f "${TMP_DIR}/model.bin" "${TARGET_MODEL_PATH}"
cp -f "${TMP_DIR}/version.json" "${LOCAL_VERSION_FILE}"
echo "[UpdateModel] Updated to model version ${REMOTE_MODEL_VER}"
