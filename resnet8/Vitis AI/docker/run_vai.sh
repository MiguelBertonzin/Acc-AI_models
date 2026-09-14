#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${VAI_DOCKER_IMAGE:-xilinx/vitis-ai-tensorflow2-cpu:ubuntu2004-3.5.0.300}"
ENV_PREFIX="/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2"
ENV_PATH="${ENV_PREFIX}/bin:/opt/vitis_ai/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV_LD_LIBRARY_PATH="${ENV_PREFIX}/lib:/opt/xilinx/xrt/lib:/usr/lib:/usr/lib/x86_64-linux-gnu"

if [[ $# -eq 0 ]]; then
  echo "Uso: $0 COMANDO [ARGUMENTOS...]" >&2
  echo "Exemplo: $0 python scripts/01_rebuild_keras2_model.py" >&2
  exit 2
fi

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env "PATH=${ENV_PATH}" \
  --env "LD_LIBRARY_PATH=${ENV_LD_LIBRARY_PATH}" \
  --env PYTHONNOUSERSITE=1 \
  --env USER=vitis-ai-user \
  --env LOGNAME=vitis-ai-user \
  --env HOME=/workspace/data/cache/home \
  --env KERAS_HOME=/workspace/data/cache/keras \
  --env TF_CPP_MIN_LOG_LEVEL=1 \
  --volume "${PROJECT_DIR}:/workspace" \
  --workdir /workspace \
  "${IMAGE}" \
  "$@"
