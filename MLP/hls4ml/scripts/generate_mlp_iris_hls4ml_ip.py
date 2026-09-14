#!/usr/bin/env python3
"""Generate, validate, synthesize, and export the MLP Iris HLS IP.

Target configuration:
  * hls4ml Vitis backend
  * ZCU104 device (xczu7ev-ffvc1156-2-e)
  * 100 MHz target clock (10 ns)
  * ap_fixed<16,6> default precision, without QKeras
  * ReuseFactor = 1
  * Strategy = Latency
  * IOType = io_parallel
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


SYSTEM_LIBSTDCXX = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")


def _ensure_compatible_libstdcxx() -> None:
    """Re-exec with the system libstdc++ required by the generated HLS library."""
    if os.environ.get("MLP_IRIS_HLS_REEXEC") == "1" or not SYSTEM_LIBSTDCXX.exists():
        return

    env = os.environ.copy()
    current = env.get("LD_PRELOAD", "")
    system_lib = str(SYSTEM_LIBSTDCXX)
    if system_lib not in current.split(":"):
        env["LD_PRELOAD"] = f"{system_lib}:{current}" if current else system_lib
    env["MLP_IRIS_HLS_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


_ensure_compatible_libstdcxx()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hls4ml  # noqa: E402
import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


HLS4ML_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = HLS4ML_ROOT / "mlp_iris_apfixed16_6_rf1_100mhz"
MODEL_PATH = ROOT / "iris_mlp_clean.h5"
SCALER_PATH = ROOT / "iris_scaler.joblib"

PROJECT_NAME = "mlp_iris"
PART = "xczu7ev-ffvc1156-2-e"
CLOCK_PERIOD_NS = 10
TARGET_FREQUENCY_MHZ = 100
DEFAULT_PRECISION = "fixed<16,6>"
REUSE_FACTOR = 1
STRATEGY = "Latency"
IO_TYPE = "io_parallel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Stop after C++ compilation and numerical validation.",
    )
    parser.add_argument(
        "--cosim",
        action="store_true",
        help="Also run RTL co-simulation (slower; synthesis and IP export always run).",
    )
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_holdout(scaler):
    features, labels = load_iris(return_X_y=True)
    _, x_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    return scaler.transform(x_test).astype(np.float32), y_test.astype(np.int64)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def configure_tool_path() -> None:
    candidates = [
        Path("/opt/Xilinx/Vitis/2024.2/bin"),
        Path("/opt/Xilinx/Vitis_HLS/2024.2/bin"),
        Path("/opt/Xilinx/Vivado/2024.2/bin"),
    ]
    additions = [str(path) for path in candidates if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join([*additions, os.environ.get("PATH", "")])


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_tool_path()

    model = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    x_test, y_test = make_holdout(scaler)
    y_keras = model.predict(x_test, verbose=0).astype(np.float32)

    input_data = output_dir / "tb_input_features.dat"
    output_data = output_dir / "tb_output_predictions.dat"
    np.savetxt(input_data, x_test, fmt="%.10g")
    np.savetxt(output_data, y_keras, fmt="%.10g")

    hls_config = hls4ml.utils.config_from_keras_model(
        model,
        granularity="model",
        backend="Vitis",
        default_precision=DEFAULT_PRECISION,
        default_reuse_factor=REUSE_FACTOR,
    )
    hls_config["Model"]["ReuseFactor"] = REUSE_FACTOR
    hls_config["Model"]["Strategy"] = STRATEGY

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        output_dir=str(output_dir),
        project_name=PROJECT_NAME,
        input_data_tb=str(input_data),
        output_data_tb=str(output_data),
        backend="Vitis",
        hls_config=hls_config,
        part=PART,
        clock_period=CLOCK_PERIOD_NS,
        io_type=IO_TYPE,
    )

    hls_model.compile()
    y_hls = np.asarray(hls_model.predict(x_test), dtype=np.float32)
    keras_classes = np.argmax(y_keras, axis=1)
    hls_classes = np.argmax(y_hls, axis=1)

    validation = {
        "samples": int(len(y_test)),
        "keras_correct": int(np.sum(keras_classes == y_test)),
        "hls_correct": int(np.sum(hls_classes == y_test)),
        "keras_accuracy": float(np.mean(keras_classes == y_test)),
        "hls_accuracy": float(np.mean(hls_classes == y_test)),
        "class_agreement_keras_hls": float(np.mean(keras_classes == hls_classes)),
        "max_absolute_output_error": float(np.max(np.abs(y_keras - y_hls))),
        "mean_absolute_output_error": float(np.mean(np.abs(y_keras - y_hls))),
        "keras_classes": keras_classes,
        "hls_classes": hls_classes,
        "expected_classes": y_test,
    }

    manifest = {
        "project_name": PROJECT_NAME,
        "model": str(MODEL_PATH),
        "scaler": str(SCALER_PATH),
        "backend": "Vitis",
        "part": PART,
        "board": "ZCU104",
        "target_frequency_mhz": TARGET_FREQUENCY_MHZ,
        "clock_period_ns": CLOCK_PERIOD_NS,
        "clock_uncertainty": "27% (hls4ml Vitis default)",
        "precision": DEFAULT_PRECISION,
        "quantization_framework": None,
        "reuse_factor": REUSE_FACTOR,
        "strategy": STRATEGY,
        "io_type": IO_TYPE,
        "softmax": "kept in the HLS model",
        "versions": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "hls4ml": hls4ml.__version__,
            "numpy": np.__version__,
            "scikit_learn_runtime": sklearn.__version__,
        },
        "hls_config": hls_config,
    }
    write_json(output_dir / "configuration_manifest.json", manifest)
    write_json(output_dir / "validation_summary.json", validation)
    np.savetxt(output_dir / "hls_predictions.dat", y_hls, fmt="%.10g")

    print(json.dumps(json_ready(validation), indent=2))
    if validation["class_agreement_keras_hls"] != 1.0:
        raise RuntimeError("HLS conversion changed at least one predicted class")

    if args.skip_build:
        print(f"Conversion and validation complete: {output_dir}")
        return 0

    report = hls_model.build(
        reset=True,
        csim=True,
        synth=True,
        cosim=args.cosim,
        validation=args.cosim,
        export=True,
        vsynth=False,
        log_to_stdout=True,
    )
    write_json(output_dir / "build_report.json", report)

    exported = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".zip", ".xo"}
            or path.name == "component.xml"
            or "export" in path.name.lower()
        )
    )
    write_json(output_dir / "exported_artifacts.json", {"files": exported})
    print(f"Synthesis and IP export complete: {output_dir}")
    print(json.dumps(json_ready(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
