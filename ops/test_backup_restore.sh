#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[TestBackup] running backup script (DB dump skipped if DATABASE_URL empty)..."
"${BASE_DIR}/ops/backup_local.sh"

LATEST_BACKUP="$(ls -1dt "${BASE_DIR}"/backups/backup_* 2>/dev/null | head -n1 || true)"
if [[ -z "${LATEST_BACKUP}" ]]; then
  echo "[TestBackup] no backup directory created"
  exit 1
fi

test -f "${LATEST_BACKUP}/checkpoints.tar.gz" || { echo "[TestBackup] checkpoints archive missing"; exit 1; }
test -f "${LATEST_BACKUP}/dataset.tar.gz" || { echo "[TestBackup] dataset archive missing"; exit 1; }

echo "[TestBackup] restore dry-run from ${LATEST_BACKUP}"
"${BASE_DIR}/ops/restore_local.sh" "${LATEST_BACKUP}"
echo "[TestBackup] ok"
