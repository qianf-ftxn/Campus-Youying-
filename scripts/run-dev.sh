#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
APP_HOST="${SERVER_HOST:-127.0.0.1}"
APP_PORT="${SERVER_PORT:-8080}"
APP_PYTHON="${PYTHON_BIN:-python3}"
if [[ -x .venv/bin/python && -z "${PYTHON_BIN:-}" ]]; then
  APP_PYTHON=.venv/bin/python
fi
# Provider/model 由 Settings 从已有 .env 读取，避免覆盖用户的 API 配置。
exec "$APP_PYTHON" -m uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT"
