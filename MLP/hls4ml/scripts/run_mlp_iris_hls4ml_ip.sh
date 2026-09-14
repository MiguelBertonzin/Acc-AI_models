#!/usr/bin/env bash
set -euo pipefail

flow_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
staging_dir="$(mktemp -d /tmp/mlp_iris_apfixed16_6_rf1_100mhz.XXXXXX)"
publish_dir="$flow_root/mlp_iris_apfixed16_6_rf1_100mhz"

python3 "$flow_root/scripts/generate_mlp_iris_hls4ml_ip.py" \
  --output-dir "$staging_dir" "$@"
mkdir -p -- "$publish_dir"
cp -a -- "$staging_dir/." "$publish_dir/"

echo "Projeto e IP publicados em: $publish_dir"
