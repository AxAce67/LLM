#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <backup_dir>"
  exit 1
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$1"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Backup directory not found: ${SRC_DIR}"
  exit 1
fi

DB_URL="${DATABASE_URL:-}"
if [[ -n "${DB_URL}" && -f "${SRC_DIR}/database.sql" ]]; then
  echo "[Restore] Restoring PostgreSQL..."
  psql "${DB_URL}" -f "${SRC_DIR}/database.sql"
else
  echo "[Restore] DB restore skipped (DATABASE_URL or database.sql missing)."
fi

if [[ -f "${SRC_DIR}/checkpoints.tar.gz" ]]; then
  echo "[Restore] Restoring checkpoints..."
  tar -xzf "${SRC_DIR}/checkpoints.tar.gz" -C "${BASE_DIR}"
fi

if [[ -f "${SRC_DIR}/dataset.tar.gz" ]]; then
  echo "[Restore] Restoring dataset..."
  tar -xzf "${SRC_DIR}/dataset.tar.gz" -C "${BASE_DIR}"
fi

echo "[Restore] Done."
