#!/usr/bin/env python3
from __future__ import annotations

import atexit
import argparse
import json
from pathlib import Path

import numpy as np

from hls_common import (
    DEFAULT_DESIGN,
    DEFAULT_MODEL,
    DEFAULT_REUSE_PLAN,
    DEFAULT_X,
    DEFAULT_Y,
    HLS_ROOT,
    collect_report_paths,
    ensure_new_output_dir,
    fixed_type,
    load_json,
    load_labels,
    make_hls_config,
    patch_vitis_2024_2_tcl,
    reference_logits,
    save_manifest,
    sync_tool_output,
    vitis_safe_path,
    write_reuse_csv,
)


def parse_args() -> argparse.Namespace:
    design = load_json(DEFAULT_DESIGN)
    parser = argparse.ArgumentParser(
        description="Gera e valida o baseline ResNet8 Resource, sem softmax em hardware."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--x", type=Path, default=DEFAULT_X)
    parser.add_argument("--y", type=Path, default=DEFAULT_Y)
    parser.add_argument("--reuse-plan", type=Path, default=DEFAULT_REUSE_PLAN)
    parser.add_argument("--out", type=Path, default=HLS_ROOT / "builds/baseline_q22_12_rf288")
    parser.add_argument("--part", default=design["part"])
    parser.add_argument("--clock", type=float, default=design["clock_period_ns"])
    parser.add_argument("--width", type=int, default=design["precision_width"])
    parser.add_argument("--integer", type=int, default=design["precision_integer"])
    parser.add_argument("--max-reuse", type=int, default=design["maximum_reuse_factor"])
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=0,
        help="Compila o C++ e valida N imagens; 0 apenas gera o projeto.",
    )
    parser.add_argument("--synth", action="store_true", help="Executa C synthesis no Vitis HLS.")
    parser.add_argument(
        "--vsynth",
        action="store_true",
        help="Também exporta o IP e executa Vivado synthesis; implica --synth.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import hls4ml
    import tensorflow as tf

    if args.validate_samples < 0:
        raise ValueError("--validate-samples não pode ser negativo.")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_dir = ensure_new_output_dir(args.out, args.force)
    tool_output_dir = vitis_safe_path(output_dir)
    precision = fixed_type(args.width, args.integer)
    atexit.register(sync_tool_output, tool_output_dir, output_dir)
    plan = {key: int(value) for key, value in load_json(args.reuse_plan).items()}
    model = tf.keras.models.load_model(model_path, compile=False)
    config, rows, softmax_mode = make_hls_config(
        model, precision, plan, args.max_reuse
    )

    (output_dir / "hls_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    write_reuse_csv(output_dir / "reuse_plan_resolved.csv", rows)
    parameters = {
        "part": args.part,
        "clock_period_ns": args.clock,
        "precision": precision,
        "io_type": "io_stream",
        "strategy": "Resource",
        "conv_implementation": "LineBuffer",
        "maximum_reuse_factor": args.max_reuse,
        "validate_samples": args.validate_samples,
        "synth": bool(args.synth or args.vsynth),
        "vsynth": args.vsynth,
    }
    save_manifest(output_dir, model_path, parameters, softmax_mode)

    print("\n=== ReuseFactor resolvido ===")
    print(f"{'Camada':18s} {'Pedido':>8s} {'Efetivo':>8s} {'Multiplicadores~':>17s}")
    for row in rows:
        print(
            f"{row['layer']:18s} {row['requested_reuse']:8d} "
            f"{row['selected_reuse']:8d} {row['estimated_parallel_multipliers']:17d}"
        )
    print(f"\nModelo:   {model_path}")
    print(f"Softmax:  {softmax_mode}")
    print(f"Precisão: {precision}")
    print(f"Saída:    {output_dir}")

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tool_output_dir),
        project_name="resnet8_resource",
        part=args.part,
        clock_period=args.clock,
        io_type="io_stream",
        backend="Vitis",
    )
    hls_model.write()
    patched = patch_vitis_2024_2_tcl(tool_output_dir)
    print(f"Patch Tcl Vitis 2024.2 aplicado: {patched}")

    if args.validate_samples:
        if not args.x.is_file() or not args.y.is_file():
            raise FileNotFoundError(
                "Dados ausentes. Execute primeiro: python scripts/02_prepare_cifar10.py"
            )
        x = np.load(args.x).astype(np.float32, copy=False)
        labels = load_labels(args.y)
        n = min(args.validate_samples, len(x), len(labels))
        x_eval = np.ascontiguousarray(x[:n], dtype=np.float32)
        labels_eval = labels[:n]
        reference = reference_logits(model, x_eval)
        hls_model.compile()
        prediction = np.asarray(hls_model.predict(x_eval)).reshape(reference.shape)
        keras_class = np.argmax(reference, axis=1)
        hls_class = np.argmax(prediction, axis=1)
        metrics = {
            "samples": n,
            "keras_hls_argmax_agreement": float(np.mean(keras_class == hls_class)),
            "keras_accuracy": float(np.mean(keras_class == labels_eval)),
            "hls_accuracy": float(np.mean(hls_class == labels_eval)),
            "logit_mae": float(np.mean(np.abs(reference - prediction))),
            "logit_max_abs_error": float(np.max(np.abs(reference - prediction))),
        }
        (output_dir / "cpp_validation.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print("\n=== Validação C++ ===")
        print(json.dumps(metrics, indent=2))

    if args.synth or args.vsynth:
        print("\nIniciando C synthesis; esta etapa pode levar bastante tempo...")
        report = hls_model.build(
            reset=False,
            csim=False,
            synth=True,
            cosim=False,
            export=args.vsynth,
            vsynth=args.vsynth,
        )
        (output_dir / "build_return.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        collect_report_paths(tool_output_dir)
    else:
        print("\nProjeto gerado. Síntese não solicitada.")


if __name__ == "__main__":
    main()

