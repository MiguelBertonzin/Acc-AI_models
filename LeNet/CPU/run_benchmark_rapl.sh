#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TURBOSTAT_LOG="/tmp/lenet_cpu_turbostat.log"
TURBOSTAT_PID_FILE="/tmp/lenet_cpu_turbostat.pid"
OUTPUT_DIR="${PROJECT_DIR}/CPU/resultados"

cleanup() {
    if [[ -r "${TURBOSTAT_PID_FILE}" ]]; then
        local turbostat_pid
        turbostat_pid="$(<"${TURBOSTAT_PID_FILE}")"
        if [[ "${turbostat_pid}" =~ ^[0-9]+$ ]]; then
            sudo kill -INT "${turbostat_pid}" 2>/dev/null || true
        fi
    fi
}

trap cleanup EXIT INT TERM

echo "O turbostat precisa de root somente para ler MSR/RAPL."
echo "O TensorFlow continuara executando como o usuario atual."
sudo -v
sudo rm -f "${TURBOSTAT_LOG}" "${TURBOSTAT_PID_FILE}"

sudo /bin/sh -c "echo \$\$ > '${TURBOSTAT_PID_FILE}'; exec /usr/bin/turbostat \
    --quiet --no-perf --Summary --debug --interval 0.1 \
    --show Time_Of_Day_Seconds,Busy%,Bzy_MHz,PkgTmp,PkgWatt,CorWatt \
    --out '${TURBOSTAT_LOG}'" &

for _ in {1..40}; do
    if [[ -s "${TURBOSTAT_LOG}" ]]; then
        sudo chmod 0644 "${TURBOSTAT_LOG}"
        break
    fi
    sleep 0.1
done

if [[ ! -r "${TURBOSTAT_LOG}" ]]; then
    echo "Falha ao iniciar o turbostat ou tornar o log legivel." >&2
    exit 1
fi

python3 "${PROJECT_DIR}/CPU/scripts/benchmark_lenet_mnist_cpu.py" \
    --model "${PROJECT_DIR}/lenet_mnist_final.h5" \
    --output-dir "${OUTPUT_DIR}" \
    --turbostat-log "${TURBOSTAT_LOG}" \
    --sample-sizes 100 1000 10000 \
    --repetitions 10 20 50 100 \
    --seed 20260825 \
    --warmup 100 \
    --baseline-seconds 5 \
    --telemetry-ms 100 \
    --restart

python3 "${PROJECT_DIR}/scripts/generate_cpu_gpu_readmes.py"

cleanup
trap - EXIT INT TERM
echo "Benchmark e coleta RAPL concluidos em ${OUTPUT_DIR}"
