#!/usr/bin/env bash
set -euo pipefail

flow_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_vai="$flow_root/docker/run_vai.sh"
host_python="${HOST_PYTHON:-python3}"

cd "$flow_root"
"$host_python" scripts/00_prepare_data.py
"$run_vai" python scripts/01_build_keras2_logits.py
"$run_vai" python scripts/02_validate_float.py
"$run_vai" python scripts/03_quantize_ptq.py
"$run_vai" python scripts/04_validate_quantized.py
"$run_vai" python scripts/05_compile_xmodel.py
"$run_vai" python scripts/06_inspect_xmodel.py
"$host_python" scripts/08_package_deploy.py
"$host_python" scripts/07_generate_manifest.py

echo "Fluxo concluído. XModel: $flow_root/artifacts/compiled/iris_mlp_zcu104_vai3_5/iris_mlp.xmodel"
