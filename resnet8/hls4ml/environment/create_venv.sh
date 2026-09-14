#!/usr/bin/env bash
set -euo pipefail

HLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HLS_ROOT}/.venv"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel
"${VENV_DIR}/bin/python" -m pip install -r "${HLS_ROOT}/environment/requirements.txt"

echo "Ambiente criado em ${VENV_DIR}"
echo "Ative com: source '${VENV_DIR}/bin/activate'"
echo "Depois valide com: python '${HLS_ROOT}/scripts/00_check_environment.py'"

