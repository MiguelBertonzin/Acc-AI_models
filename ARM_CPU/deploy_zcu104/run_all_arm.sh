#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke) DEFAULT_INFERENCES=30; DEFAULT_WARMUP=10 ;;
  full) DEFAULT_INFERENCES=10000; DEFAULT_WARMUP=200 ;;
  *) echo "Uso: $0 [smoke|full]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INFERENCES="${INFERENCES:-$DEFAULT_INFERENCES}"
WARMUP="${WARMUP:-$DEFAULT_WARMUP}"
REPETITIONS="${REPETITIONS:-1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${RESULT_DIR:-results/zcu104_${STAMP}}"
mkdir -p "$RESULT_DIR"

{
  date -u
  uname -a
  python3 --version
  python3 scripts/check_board.py
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true
  cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true
} > "$RESULT_DIR/environment_before.txt" 2>&1

for repetition in $(seq 1 "$REPETITIONS"); do
  for network in mlp lenet resnet8; do
    for precision in fp32 int8; do
      for threads in 1 4; do
        [[ "$threads" == "1" ]] && cpus="0" || cpus="0-3"
        name="${network}_${precision}_t${threads}_run${repetition}"
        echo "Executando $name: $INFERENCES inferências, warm-up $WARMUP"
        taskset -c "$cpus" python3 scripts/benchmark_arm.py \
          --network "$network" --precision "$precision" --threads "$threads" \
          --warmup "$WARMUP" --inferences "$INFERENCES" \
          --output "$RESULT_DIR/${name}.json" \
          > "$RESULT_DIR/${name}.log" 2>&1
      done
    done
  done
done

{
  date -u
  uptime
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true
  cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null || true
  sha256sum -c SHA256SUMS.txt
} > "$RESULT_DIR/environment_after.txt" 2>&1

echo "Campanha concluída em: $RESULT_DIR"
