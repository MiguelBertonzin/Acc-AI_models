#!/usr/bin/env python3
"""Gera vetores golden de placa diretamente das transações aprovadas pelo RTL co-sim."""
from __future__ import annotations

import csv, hashlib, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "mlp_iris_apfixed16_6_rf1_100mhz"
SIM = BUILD / "mlp_iris_prj/solution1/sim/tv"
OUT = ROOT / "golden"
FRAC = 10


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transactions(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"\[\[transaction\]\]\s+\d+\s+(0x[0-9a-fA-F]+)\s+\[\[/transaction\]\]", text)
    if not blocks:
        raise RuntimeError(f"Nenhuma transação em {path}")
    return [int(value, 16) for value in blocks]


def signed16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def main() -> None:
    input_path = SIM / "cdatafile/c.mlp_iris.autotvin_features.dat"
    output_paths = [SIM / f"rtldatafile/rtl.mlp_iris.autotvout_layer7_out_{i}.dat" for i in range(3)]
    packed = np.asarray(transactions(input_path), dtype=np.uint64)
    raw_inputs = np.asarray([[signed16(int(word) >> (16*i)) for i in range(4)] for word in packed], dtype=np.int16)
    raw_outputs = np.asarray([[signed16(transactions(path)[row]) for path in output_paths] for row in range(len(packed))], dtype=np.int16)
    normalized = np.loadtxt(BUILD / "tb_input_features.dat", dtype=np.float32)
    expected = np.asarray(json.loads((BUILD / "validation_summary.json").read_text())["expected_classes"], dtype=np.int64)
    predicted = np.argmax(raw_outputs, axis=1).astype(np.int64)
    if len(packed) != 30 or normalized.shape != (30,4) or raw_outputs.shape != (30,3):
        raise RuntimeError("Dimensão golden inesperada")
    if int(np.sum(predicted == expected)) != 29:
        raise RuntimeError("RTL golden não preservou 29/30")
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / "iris_holdout_hls_apfixed16_6.npz"
    np.savez(npz, normalized_features=normalized, features_packed_u64=packed,
             input_fixed_raw_i16=raw_inputs, output_fixed_raw_i16=raw_outputs,
             output_values=raw_outputs.astype(np.float32)/(2**FRAC),
             predictions=predicted, labels=expected)
    csv_path = OUT / "iris_holdout_hls_apfixed16_6.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["sample","packed_input_hex",*[f"x{i}_raw_i16" for i in range(4)],*[f"y{i}_raw_i16" for i in range(3)],"prediction","label"])
        for i in range(30):
            w.writerow([i,f"0x{int(packed[i]):016x}",*map(int,raw_inputs[i]),*map(int,raw_outputs[i]),int(predicted[i]),int(expected[i])])
    readme = OUT / "README.md"
    readme.write_text("""# Pacote golden HLS4ML para a ZCU104

Estes 30 vetores vêm diretamente das transações aprovadas no RTL co-sim do IP, e são a referência funcional da placa.

- Tipo: `ap_fixed<16,6>` (`AP_TRN`, `AP_WRAP`), 10 bits fracionários.
- Entrada empacotada: `x0[15:0]`, `x1[31:16]`, `x2[47:32]`, `x3[63:48]`.
- Saída: três valores crus signed-16; valor real = `raw / 1024`.
- Resultado esperado: 29/30 corretas e classes exatamente iguais a `predictions`.
- `features_packed_u64` deve ser escrito no registrador/stream de entrada sem nova quantização.

Na placa, teste primeiro todos os vetores deste pacote e só libere o benchmark se a concordância for 30/30 com `predictions`. O único erro esperado de classificação é a amostra cujo rótulo difere da predição; não ajuste o resultado para escondê-lo.
""", encoding="utf-8")
    manifest = {"status":"passed","source":"RTL co-simulation transaction files","samples":30,"format":{"precision":"ap_fixed<16,6>","fractional_bits":10,"quantization":"AP_TRN","overflow":"AP_WRAP","input_packing":"x0[15:0], x1[31:16], x2[47:32], x3[63:48]"},"validation":{"correct":29,"accuracy":29/30,"golden_class_agreement":1.0},"source_sha256":{"input_transactions":sha(input_path),**{f"output_{i}_transactions":sha(p) for i,p in enumerate(output_paths)}},"files":[]}
    for p in (npz,csv_path,readme): manifest["files"].append({"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)})
    mp=OUT/"MANIFEST.json"; mp.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (OUT/"SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted([npz,csv_path,readme,mp])),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__ == "__main__": main()
