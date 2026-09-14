#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# LeNet / hls4ml / ZCU104 - cópia e arquivamento da campanha
# Executar NO PC.
# Data real da campanha: 2026-09-10
# ============================================================

BASE="/home/miguel/Downloads/Plano testes TCC/LeNet/hls4ml"
NAME="2026-09-10_lenet_mnist_hls4ml_zcu104"
PARENT="$BASE/coletas_zcu104_lenet"
DEST="$PARENT/$NAME"

BOARD="xilinx@192.168.2.99"
REMOTE="/home/xilinx/jupyter_notebooks/lenet_mnist"

echo "============================================================"
echo "LeNet ZCU104 -> PC"
echo "============================================================"
echo "BOARD : $BOARD"
echo "REMOTE: $REMOTE"
echo "DEST  : $DEST"
echo

command -v ssh >/dev/null || { echo "ERRO: ssh não encontrado."; exit 1; }
command -v rsync >/dev/null || { echo "ERRO: rsync não encontrado. Instale com: sudo apt install rsync"; exit 1; }
command -v sha256sum >/dev/null || { echo "ERRO: sha256sum não encontrado."; exit 1; }

mkdir -p "$DEST/board_workspace"
mkdir -p "$DEST/pc_reference"
mkdir -p "$DEST/pc_vivado"

echo "[1/9] Testando SSH..."
ssh "$BOARD" "test -d '$REMOTE' && echo 'OK: workspace remoto encontrado'"

echo "[2/9] Registrando ambiente da placa..."
{
    echo "ACTUAL_EXPERIMENT_DATE=2026-09-10"
    echo "NOTE=ZCU104 clock was incorrect; RUN_ID 20250504_164021 is not the real experiment date."
    echo
    ssh "$BOARD" '
        echo "===== DATE REPORTED BY BOARD ====="
        date
        echo
        echo "===== UNAME ====="
        uname -a
        echo
        echo "===== OS ====="
        cat /etc/os-release 2>/dev/null || true
        echo
        echo "===== PYTHON ====="
        python3 --version
        echo
        echo "===== CPUFREQ ====="
        for p in /sys/devices/system/cpu/cpufreq/policy*; do
            echo "--- $p"
            for f in scaling_governor scaling_cur_freq scaling_min_freq scaling_max_freq cpuinfo_min_freq cpuinfo_max_freq scaling_setspeed; do
                if [ -f "$p/$f" ]; then
                    printf "%s=" "$f"
                    cat "$p/$f"
                fi
            done
        done
    '
} > "$DEST/BOARD_ENVIRONMENT.txt"

echo "[3/9] Gerando lista de arquivos remota..."
ssh "$BOARD" "
    cd '$REMOTE' &&
    find . -type f -printf '%p\t%s bytes\n' |
    sort
" > "$DEST/BOARD_FILE_LIST.txt"

echo "[4/9] Gerando manifesto SHA256 remoto..."
ssh "$BOARD" "
    cd '$REMOTE' &&
    find . -type f -print0 |
    sort -z |
    xargs -0 sha256sum
" > "$DEST/SHA256SUMS_BOARD.txt"

echo "[5/9] Copiando workspace completo da ZCU104..."
rsync -avh \
    --info=progress2 \
    --partial \
    "$BOARD:$REMOTE/" \
    "$DEST/board_workspace/"

echo "[6/9] Verificando SHA256 da cópia..."
(
    cd "$DEST/board_workspace"
    sha256sum -c "../SHA256SUMS_BOARD.txt"
) | tee "$DEST/SHA256_VERIFY.txt"

if grep -q "FAILED" "$DEST/SHA256_VERIFY.txt"; then
    echo "ERRO: pelo menos um arquivo falhou na verificação SHA256."
    exit 2
fi

echo "OK: hashes da ZCU104 verificados."

echo "[7/9] Copiando modelo e snapshot do deployment local..."
MODEL="/home/miguel/Downloads/Plano testes TCC/LeNet/lenet_mnist_final.h5"
DEPLOY="$BASE/hardware/deploy/lenet_zcu104_q22_12_rf5_50_64_60_42_100mhz"

if [ -f "$MODEL" ]; then
    cp -av "$MODEL" "$DEST/pc_reference/"
else
    echo "AVISO: modelo não encontrado em $MODEL"
fi

if [ -d "$DEPLOY" ]; then
    mkdir -p "$DEST/pc_reference/deploy_snapshot"
    rsync -avh "$DEPLOY/" "$DEST/pc_reference/deploy_snapshot/"
else
    echo "AVISO: deployment local não encontrado em $DEPLOY"
fi

(
    cd "$DEST/pc_reference"
    find . -type f -print0 | sort -z | xargs -0 -r sha256sum
) > "$DEST/PC_REFERENCE_SHA256.txt"

echo "[8/9] Criando snapshot opcional do projeto Vivado..."
shopt -s nullglob
VIVADO_ITEMS=(/home/miguel/LeNet-tcc3-08_09*)
shopt -u nullglob

if [ "${#VIVADO_ITEMS[@]}" -gt 0 ]; then
    (
        cd /home/miguel
        tar -czf \
            "$DEST/pc_vivado/LeNet-tcc3-08_09_Vivado_full.tar.gz" \
            LeNet-tcc3-08_09*
    )
    sha256sum \
        "$DEST/pc_vivado/LeNet-tcc3-08_09_Vivado_full.tar.gz" \
        > "$DEST/pc_vivado/LeNet-tcc3-08_09_Vivado_full.tar.gz.sha256"
else
    echo "AVISO: nenhum arquivo /home/miguel/LeNet-tcc3-08_09* encontrado."
fi

# Se os READMEs já estiverem no BASE, preserva dentro do pacote.
for f in \
    "$BASE/README_FINAL_LENET_ZCU104_HLS4ML_COLETA_COMPLETA.md" \
    "$BASE/README_FINAL_LENET_ZCU104_HLS4ML_COLETA_COMPLETA.txt"
do
    if [ -f "$f" ]; then
        cp -av "$f" "$DEST/"
    fi
done

echo "[9/9] Criando pacote final tar.gz..."
mkdir -p "$PARENT"

(
    cd "$PARENT"
    tar -czf "${NAME}.tar.gz" "$NAME"
    sha256sum "${NAME}.tar.gz" > "${NAME}.tar.gz.sha256"
    sha256sum -c "${NAME}.tar.gz.sha256"
)

echo
echo "============================================================"
echo "ARQUIVAMENTO CONCLUÍDO"
echo "============================================================"
echo "Pasta:"
echo "  $DEST"
echo
echo "Pacote:"
echo "  $PARENT/${NAME}.tar.gz"
echo
echo "Hash:"
echo "  $PARENT/${NAME}.tar.gz.sha256"
echo
echo "Verificação da cópia da placa:"
echo "  $DEST/SHA256_VERIFY.txt"
echo
echo "Não apague a ZCU104 antes de revisar SHA256_VERIFY.txt."
