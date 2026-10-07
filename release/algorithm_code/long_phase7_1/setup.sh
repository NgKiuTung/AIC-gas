#!/usr/bin/env bash
# 私有AIC赛事研究：只建立项目环境，宿主Torch及驱动保持不变。
set -euo pipefail
cd "$(dirname "$0")"
HOST_PYTHON="${GAS_HOST_PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
    "$HOST_PYTHON" -m venv --system-site-packages .venv
fi
.venv/bin/python tools/install_missing.py
