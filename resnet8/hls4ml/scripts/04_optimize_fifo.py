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
    HLS_ROOT,
    audit_fifo_rtl,
    collect_report_paths,
    ensure_new_output_dir,
    fixed_type,
    load_json,
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
        description="Executa profiling RTL e otimização de profundidade FIFO da ResNet8."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--x", type=Path, default=DEFAULT_X)
    parser.add_argument("--reuse-plan", type=Path, default=DEFAULT_REUSE_PLAN)
    parser.add_argument("--out", type=Path, default=HLS_ROOT / "builds/fifo_opt_q22_12_rf288")
    parser.add_argument("--part", default=design["part"])
    parser.add_argument("--clock", type=float, default=design["clock_period_ns"])
    parser.add_argument("--width", type=int, default=design["precision_width"])
    parser.add_argument("--integer", type=int, default=design["precision_integer"])
    parser.add_argument("--max-reuse", type=int, default=design["maximum_reuse_factor"])
    parser.add_argument(
        "--profiling-depth", type=int, default=design["profiling_fifo_depth"]
    )
    parser.add_argument(
        "--tb-samples",
        type=int,
        default=design["minimum_fifo_test_samples"],
        help="O flow Vitis requer pelo menos duas chamadas do top.",
    )
    parser.add_argument(
        "--vsynth", action="store_true", help="Exporta IP e executa Vivado synthesis final."
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def check_fifo_profile(output_dir: Path, profiling_depth: int) -> dict:
    path = output_dir / "fifo_depths.json"
    if not path.is_file():
        raise FileNotFoundError(f"O flow não gerou {path}.")
    depths = json.loads(path.read_text(encoding="utf-8"))
    if not depths:
        raise RuntimeError("O relatório FIFO está vazio.")
    maximum = max(int(item["optimized"]) for item in depths.values())
    saturated = [
        name
        for name, item in depths.items()
        if int(item["optimized"]) >= profiling_depth
    ]
    summary = {
        "fifo_count": len(depths),
        "profiling_depth": profiling_depth,
        "maximum_optimized_depth": maximum,
        "saturated_fifos": saturated,
        "valid": not saturated,
    }
    (output_dir / "fifo_profile_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if saturated:
        raise RuntimeError(
            "Profiling FIFO saturado; repita com --profiling-depth maior. "
            f"Canais: {saturated}"
        )
    return summary


def main() -> None:
    args = parse_args()
    import hls4ml
    import tensorflow as tf

    if args.tb_samples < 2:
        raise ValueError(
            "--tb-samples deve ser >= 2: o otimizador Vitis exige ao menos duas chamadas do top."
        )
    if args.profiling_depth < 2:
        raise ValueError("--profiling-depth deve ser >= 2.")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not args.x.is_file():
        raise FileNotFoundError(
            "CIFAR-10 ausente. Execute primeiro: python scripts/02_prepare_cifar10.py"
        )
    output_dir = ensure_new_output_dir(args.out, args.force)
    tool_output_dir = vitis_safe_path(output_dir)
    tb_dir = output_dir / "testbench"
    atexit.register(sync_tool_output, tool_output_dir, output_dir)
    tb_dir.mkdir()
    precision = fixed_type(args.width, args.integer)
    plan = {key: int(value) for key, value in load_json(args.reuse_plan).items()}
    model = tf.keras.models.load_model(model_path, compile=False)

    x_all = np.load(args.x).astype(np.float32, copy=False)
    n_tb = min(args.tb_samples, len(x_all))
    x_tb = np.ascontiguousarray(x_all[:n_tb], dtype=np.float32)
    y_tb = reference_logits(model, x_tb)
    input_tb_path = tb_dir / "fifo_tb_input.npy"
    output_tb_path = tb_dir / "fifo_tb_logits.npy"
    np.save(input_tb_path, x_tb)
    np.save(output_tb_path, y_tb)

    config, rows, softmax_mode = make_hls_config(
        model, precision, plan, args.max_reuse
    )
    config["Flows"] = ["vitis:fifo_depth_optimization"]
    fifo_optimizer = hls4ml.model.optimizer.get_optimizer(
        "vitis:fifo_depth_optimization"
    )
    fifo_optimizer.configure(profiling_fifo_depth=args.profiling_depth)
    (output_dir / "hls_config_fifo_opt.json").write_text(
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
        "profiling_fifo_depth": args.profiling_depth,
        "tb_samples": n_tb,
        "vsynth": args.vsynth,
    }
    save_manifest(output_dir, model_path, parameters, softmax_mode)

    print("=== FIFO depth optimization ===")
    print(f"Modelo:           {model_path}")
    print(f"Softmax:          {softmax_mode}")
    print(f"Reuse máximo:     {args.max_reuse}")
    print(f"FIFO de profiling:{args.profiling_depth}")
    print(f"Chamadas do top:  {n_tb}")
    print("A conversão abaixo já executa C synthesis + RTL co-simulation de profiling.")

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(tool_output_dir),
        project_name="resnet8_resource_fifo_opt",
        input_data_tb=str(input_tb_path.resolve()),
        output_data_tb=str(output_tb_path.resolve()),
        part=args.part,
        clock_period=args.clock,
        io_type="io_stream",
        backend="Vitis",
    )

    profile = check_fifo_profile(tool_output_dir, args.profiling_depth)
    print("\nProfiling concluído:")
    print(json.dumps(profile, indent=2))

    # Grava novamente o projeto, agora com as profundidades otimizadas.
    hls_model.write()
    patched = patch_vitis_2024_2_tcl(tool_output_dir)
    print(f"Patch Tcl Vitis 2024.2 aplicado ao projeto final: {patched}")

    print("\nIniciando C synthesis final com os FIFOs reescritos...")
    report = hls_model.build(
        reset=False,
        csim=True,
        synth=True,
        cosim=False,
        export=args.vsynth,
        vsynth=args.vsynth,
    )
    (output_dir / "build_return.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    collect_report_paths(tool_output_dir)
    rtl_audit = audit_fifo_rtl(tool_output_dir)
    print("\nAuditoria RTL:")
    print(json.dumps(rtl_audit, indent=2))
    if rtl_audit["top_instantiates_d4096"]:
        print(
            "ATENÇÃO: o top final ainda instancia FIFO d4096; não promova este IP."
        )


if __name__ == "__main__":
    main()

