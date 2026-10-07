#!/usr/bin/env bash
# 私有AIC研究：固定项目解释器，失败立即退出，不重装系统依赖。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  printf '%s\n' '项目环境不存在；先执行 bash setup.sh。' >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" -u "$ROOT/run.py" research "$@"
