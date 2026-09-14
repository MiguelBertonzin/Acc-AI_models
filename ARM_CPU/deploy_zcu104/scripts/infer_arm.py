#!/usr/bin/env python3
"""Executa uma inferência CPU-only na ZCU104 usando um modelo TFLite."""

import argparse
import json
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from model_specs import ROOT, SPECS, prepare_sample

LABELS = {
    "mlp": ["setosa", "versicolor", "virginica"],
    "lenet": [str(value) for value in range(10)],
    "resnet8": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
}

def quantize(value, detail):
    scale, zero = detail["quantization"]
    return np.clip(np.rint(value / scale + zero), -128, 127).astype(np.int8)

def dequantize(value, detail):
    scale, zero = detail["quantization"]
    return (value.astype(np.float32) - zero) * scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, choices=SPECS)
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--threads", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--index", type=int, default=0, help="Índice no conjunto de teste local")
    parser.add_argument("--features", nargs=4, type=float, metavar=("SEPAL_L", "SEPAL_W", "PETAL_L", "PETAL_W"), help="Atributos Iris brutos; exclusivo da MLP")
    args = parser.parse_args()

    spec = SPECS[args.network]
    model_path = ROOT / f"models/tflite/{args.network}_{args.precision}.tflite"
    interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=args.threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    expected = None
    if args.features is not None:
        if args.network != "mlp":
            parser.error("--features é exclusivo da MLP")
        scaler = np.load(ROOT / "data/mlp/iris_scaler_params.npz")
        raw = np.asarray(args.features, dtype=np.float32)
        value = ((raw - scaler["mean"]) / scaler["scale"]).reshape(1, 4)
        source = {"raw_features": raw.tolist()}
    else:
        data = np.load(spec["test"])
        if not 0 <= args.index < len(data[spec["input_key"]]):
            parser.error("índice fora do conjunto de teste")
        value = prepare_sample(spec, data[spec["input_key"]][args.index])
        expected = int(data[spec["labels_key"]][args.index])
        source = {"test_index": args.index, "expected_class": expected}

    value = quantize(value, input_detail) if input_detail["dtype"] == np.int8 else value.astype(input_detail["dtype"], copy=False)
    interpreter.set_tensor(input_detail["index"], value)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_detail["index"])
    output = dequantize(raw_output, output_detail) if output_detail["dtype"] == np.int8 else raw_output
    predicted = int(np.argmax(output[0]))
    result = {
        "network": spec["name"], "precision": args.precision, "threads": args.threads,
        **source, "predicted_class": predicted, "predicted_label": LABELS[args.network][predicted],
        "correct": None if expected is None else predicted == expected,
        "output": output[0].astype(float).tolist(),
        "raw_output": raw_output[0].astype(int).tolist() if raw_output.dtype == np.int8 else None,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
