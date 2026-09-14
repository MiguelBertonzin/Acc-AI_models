#!/usr/bin/env python3
"""Gera o baseline hls4ml e valida o C++ nas 10.000 imagens MNIST."""

from __future__ import annotations

import atexit
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path


SYSTEM_LIBSTDCXX = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
if os.environ.get("LENET_HLS_REEXEC") != "1" and SYSTEM_LIBSTDCXX.exists():
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(SYSTEM_LIBSTDCXX)
    env["LENET_HLS_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hls4ml
import keras
import numpy as np
import tensorflow as tf
from hls4ml.backends.backend import get_backend


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MODEL = PROJECT / "lenet_mnist_final.h5"
DEFAULT_DESIGN = ROOT / "configs/design_initial_zcu104.json"
DEFAULT_REUSE = ROOT / "configs/reuse_initial_balanced.json"
DATA = ROOT / "data/mnist_test_uint8.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_precision(value, precision: str):
    if isinstance(value, dict):
        return {key: replace_precision(item, precision) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_precision(item, precision) for item in value]
    return precision


def wilson(correct: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = correct / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [float(center - radius), float(center + radius)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-config", default=str(DEFAULT_DESIGN))
    parser.add_argument("--reuse-config", default=str(DEFAULT_REUSE))
    parser.add_argument("--output-name", default="baseline_q22_12_resource_stream_balanced")
    parser.add_argument("--project-name", default="lenet_mnist_hls")
    args = parser.parse_args()
    design_path = Path(args.design_config).resolve()
    reuse_path = Path(args.reuse_config).resolve()
    output = ROOT / "builds" / args.output_name
    for path in (MODEL, design_path, reuse_path, DATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"preservando build existente: {output}")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    reuse_payload = json.loads(reuse_path.read_text(encoding="utf-8"))
    reuse = reuse_payload.get("reuse", reuse_payload)
    model = keras.models.load_model(MODEL, compile=False)
    model_config = model.get_config()
    input_layers = [item for item in model_config.get("layers", []) if item.get("class_name") == "InputLayer"]
    if input_layers and input_layers[0].get("config", {}).get("name") == "input":
        input_layers[0]["config"]["name"] = "input_layer"
        renamed_model = model.__class__.from_config(model_config)
        renamed_model.set_weights(model.get_weights())
        model = renamed_model
    if keras.activations.serialize(model.layers[-1].activation) != "linear":
        raise RuntimeError("modelo deve permanecer sem Softmax")
    precision = design["precision"]
    config = hls4ml.utils.config_from_keras_model(
        model, granularity="name", backend="Vitis",
        default_precision=precision, default_reuse_factor=16,
    )
    config["Model"]["Precision"] = precision
    config["Model"]["ReuseFactor"] = 16
    config["Model"]["Strategy"] = "Resource"
    config["Model"]["TraceOutput"] = False
    for layer_config in config.get("LayerName", {}).values():
        layer_config["Trace"] = False
        if "Precision" in layer_config:
            layer_config["Precision"] = replace_precision(layer_config["Precision"], precision)

    backend = get_backend("Vitis")
    rows = []
    for layer in model.layers:
        if layer.name not in reuse:
            continue
        kernel = layer.get_weights()[0]
        n_in, n_out = int(np.prod(kernel.shape[:-1])), int(kernel.shape[-1])
        valid = [int(value) for value in backend.get_valid_reuse_factors(n_in, n_out)]
        selected = int(reuse[layer.name])
        if selected not in valid:
            raise RuntimeError(f"ReuseFactor invalido: {layer.name}={selected}; validos={valid}")
        layer_config = config["LayerName"][layer.name]
        layer_config["ReuseFactor"] = selected
        layer_config["Strategy"] = "Resource"
        if layer.__class__.__name__ == "Conv2D":
            layer_config["ConvImplementation"] = "LineBuffer"
            layer_config["ParallelizationFactor"] = 1
        rows.append({
            "layer": layer.name, "class": layer.__class__.__name__, "n_in": n_in, "n_out": n_out,
            "multiplications": n_in * n_out, "reuse_factor": selected,
            "estimated_multiplier_limit": int(np.ceil(n_in * n_out / min(n_in, selected))),
        })

    tmp_root = Path(tempfile.mkdtemp(prefix="lenet_hls4ml_"))
    stage = tmp_root / output.name
    def cleanup() -> None:
        shutil.rmtree(tmp_root, ignore_errors=True)
    atexit.register(cleanup)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(stage), project_name=args.project_name,
        part=design["part"], clock_period=float(design["clock_period_ns"]),
        io_type=design["io_type"], backend=design["backend"],
    )
    hls_model.write()
    tcl = stage / "build_prj.tcl"
    if tcl.is_file():
        original = tcl.read_text(encoding="utf-8")
        patched = "\n".join(line for line in original.splitlines() if "config_array_partition -maximum_size" not in line) + "\n"
        if patched != original:
            (stage / "build_prj.original.tcl").write_text(original, encoding="utf-8")
            tcl.write_text(patched, encoding="utf-8")
    hls_model.compile()

    raw = np.load(DATA)
    x = np.ascontiguousarray((raw["images"].astype(np.float32) / 255.0)[..., np.newaxis])
    labels = raw["labels"].reshape(-1).astype(np.int64)
    reference = model.predict(x, batch_size=256, verbose=0).astype(np.float32)
    prediction = np.asarray(hls_model.predict(x), dtype=np.float32).reshape(reference.shape)
    ref_class, hls_class = np.argmax(reference, axis=1), np.argmax(prediction, axis=1)
    ref_correct = int(np.sum(ref_class == labels))
    hls_correct = int(np.sum(hls_class == labels))
    agreement = int(np.sum(ref_class == hls_class))
    error = np.abs(reference - prediction)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, hls_class), 1)
    status = "passed" if agreement / len(labels) >= 0.999 and hls_correct >= ref_correct - 10 else "failed"
    validation = {
        "status": status, "samples": int(len(labels)), "keras_correct": ref_correct,
        "keras_accuracy_percent": 100.0 * ref_correct / len(labels), "hls_correct": hls_correct,
        "hls_accuracy_percent": 100.0 * hls_correct / len(labels), "hls_wilson_95": wilson(hls_correct, len(labels)),
        "accuracy_delta_hls_minus_keras_pp": 100.0 * (hls_correct - ref_correct) / len(labels),
        "keras_hls_argmax_agreement": agreement, "keras_hls_argmax_agreement_percent": 100.0 * agreement / len(labels),
        "logit_mean_absolute_error": float(error.mean()), "logit_max_absolute_error": float(error.max()),
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "acceptance": {"minimum_argmax_agreement_percent": 99.9, "maximum_accuracy_loss_images": 10},
    }

    shutil.copytree(stage, output)
    (output / "hls_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    with (output / "reuse_plan_resolved.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / "cpp_predictions_10000.npz", keras_logits=reference, hls_logits=prediction, labels=labels, keras_predictions=ref_class, hls_predictions=hls_class)
    (output / "cpp_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": status, "model": str(MODEL), "model_sha256": sha256(MODEL),
        "design": design, "design_config": str(design_path), "reuse": reuse,
        "reuse_config": str(reuse_path), "resolved_layers": rows,
        "estimated_multiplier_limit_total": sum(row["estimated_multiplier_limit"] for row in rows),
        "software": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__, "hls4ml": hls4ml.__version__, "numpy": np.__version__},
        "note": "C++ validation complete; Vitis HLS synthesis and IP export not executed yet",
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(f"Build C++ preservado em {output}")
    if status != "passed":
        raise RuntimeError("baseline numerico reprovado")


if __name__ == "__main__":
    main()
