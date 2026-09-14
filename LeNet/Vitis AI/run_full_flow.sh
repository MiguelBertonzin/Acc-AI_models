#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 scripts/00_export_keras3_reference.py
docker/run_vai.sh python scripts/01_rebuild_keras2_model.py
docker/run_vai.sh python scripts/02_prepare_mnist.py
docker/run_vai.sh python scripts/03_validate_float_model.py
docker/run_vai.sh python scripts/04_quantize_ptq.py
docker/run_vai.sh python scripts/05_validate_quantized_model.py
docker/run_vai.sh python scripts/06_compile_xmodel.py
docker/run_vai.sh python scripts/07_inspect_xmodel.py
docker/run_vai.sh python scripts/08_package_deploy.py
docker/run_vai.sh python scripts/09_generate_manifest.py

echo "Fluxo concluido. XModel em artifacts/compiled/lenet_mnist_zcu104_vai3_5/"
