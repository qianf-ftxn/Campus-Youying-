#!/usr/bin/env bash
set -euo pipefail

# Finder 打开的 Terminal 不一定继承交互 shell 的 Homebrew / Docker 路径。
export PATH="/opt/homebrew/bin:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  APP_PYTHON=""
  for candidate in python3.11 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      APP_PYTHON="$(command -v "$candidate")"
      break
    fi
  done
  if [[ -z "$APP_PYTHON" ]]; then
    echo "未找到 Python 3.11 / 3.12。请先安装 Python 3.11 后重新双击。"
    exit 1
  fi
  "$APP_PYTHON" -m venv .venv
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "已创建 .env，请按所用模型填写配置；已有配置不会被覆盖。"
fi

if ! .venv/bin/python -m pip check >/dev/null 2>&1 || \
   ! .venv/bin/python -c 'import fastapi, uvicorn, sqlalchemy, pymysql, redis, pydantic_settings, chromadb, multipart, httpx, openpyxl, pypdf, mcp' >/dev/null 2>&1; then
  echo "正在安装或补齐项目依赖……"
  .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/python scripts/launch.py "$@"
