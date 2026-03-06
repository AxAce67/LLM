#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BASE_DIR}/backups"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BACKUP_DIR}/backup_${TS}"
mkdir -p "${OUT_DIR}"

DB_URL="${DATABASE_URL:-}"
if [[ -n "${DB_URL}" ]]; then
  echo "[Backup] Dumping PostgreSQL..."
  pg_dump "${DB_URL}" > "${OUT_DIR}/database.sql"
else
  echo "[Backup] DATABASE_URL is empty. Skipping DB dump."
fi

echo "[Backup] Archiving checkpoints and dataset..."
tar -czf "${OUT_DIR}/checkpoints.tar.gz" -C "${BASE_DIR}" checkpoints || true
tar -czf "${OUT_DIR}/dataset.tar.gz" -C "${BASE_DIR}" dataset || true

echo "[Backup] Done: ${OUT_DIR}"
