#!/usr/bin/env python3
"""Generate LeNet H1 with only the requested global RF changed."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import zipfile
from ip_audit import audit_ip

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

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "lenet_mnist_final.h5"
DATA = ROOT / "data/mnist_test_uint8.npz"
BASELINE = ROOT.parents[1] / "LeNet/hls4ml/builds/cap64_q22_12_rate_balanced"
TOP = "lenet_mnist_cap64_hls"

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")

def wilson(correct, total):
    z = 1.959963984540054
    p = correct / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    radius = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [float(center-radius), float(center+radius)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-factor", type=int, choices=(32,64,128), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    rf = args.reuse_factor
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = os.pathsep.join(["/opt/Xilinx/Vitis/2024.2/bin", "/opt/Xilinx/Vitis_HLS/2024.2/bin", "/opt/Xilinx/Vivado/2024.2/bin", os.environ["PATH"]])
    design = json.loads((ROOT / "reference/design.json").read_text())
    assert sha(MODEL) == design["model_sha256"]
    config = copy.deepcopy(json.loads((ROOT / "reference/hls_config.json").read_text()))
    config["Model"]["ReuseFactor"] = rf
    for layer_config in config["LayerName"].values():
        layer_config.pop("ReuseFactor", None)
    write_json(out / "hls_config_requested.json", config)
    model = keras.models.load_model(MODEL, compile=False)
    model_config = model.get_config()
    input_layers = [x for x in model_config.get("layers",[]) if x.get("class_name") == "InputLayer"]
    if input_layers and input_layers[0]["config"].get("name") == "input":
        input_layers[0]["config"]["name"] = "input_layer"
        renamed = model.__class__.from_config(model_config)
        renamed.set_weights(model.get_weights())
        model = renamed
    assert keras.activations.serialize(model.layers[-1].activation) == "linear"
    raw = np.load(DATA)
    x = np.ascontiguousarray((raw["images"].astype(np.float32)/255.0)[...,np.newaxis])
    labels = raw["labels"].reshape(-1).astype(np.int64)
    assert x.shape == (10000,28,28,1)
    indices = np.array([np.flatnonzero(labels == label)[0] for label in range(10)], dtype=np.int64)
    reference = model.predict(x, batch_size=256, verbose=0).astype(np.float32)
    np.savetxt(out / "tb_input_features.dat", x[indices].reshape(10,-1), fmt="%.10g")
    np.savetxt(out / "tb_output_predictions.dat", reference[indices], fmt="%.10g")
    write_json(out / "rtl_test_subset.json", {"selection":"first test image of each class, in class order 0..9", "indices":indices.tolist(), "labels":labels[indices].tolist(), "samples":10})
    print(f"RF{rf}: converting model", flush=True)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config,
        output_dir=str(out), project_name=TOP, part=design["part"],
        clock_period=design["clock_period_ns"], io_type=design["io_type"], backend=design["backend"],
        input_data_tb=str(out / "tb_input_features.dat"), output_data_tb=str(out / "tb_output_predictions.dat"))
    backend = get_backend("Vitis")
    graph = {layer.name:layer for layer in hls_model.get_layers()}
    effective = [{"name":layer.name,"class":layer.__class__.__name__,
        "reuse_factor":layer.get_attr("reuse_factor"),"strategy":layer.get_attr("strategy")} for layer in hls_model.get_layers()]
    write_json(out / "effective_layers.json",effective)
    resolved = []
    for layer in model.layers:
        if layer.__class__.__name__ not in {"Dense","Conv2D"}:
            continue
        kernel=layer.get_weights()[0]
        n_in,n_out=int(np.prod(kernel.shape[:-1])),int(kernel.shape[-1])
        valid=[int(v) for v in backend.get_valid_reuse_factors(n_in,n_out)]
        actual=int(graph[layer.name].get_attr("reuse_factor"))
        closest=int(backend.get_closest_reuse_factor(valid,rf))
        assert actual == closest and graph[layer.name].get_attr("strategy") == "resource"
        resolved.append({"layer":layer.name,"n_in":n_in,"n_out":n_out,
            "requested_rf":rf,"effective_rf":actual,"strategy":"resource","valid_rf":valid})
    write_json(out / "reuse_resolved.json",resolved)
    with (out / "reuse_resolved.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(resolved[0]))
        writer.writeheader(); writer.writerows(resolved)
    hls_model.write()
    source=(out / "firmware" / f"{TOP}.cpp").read_text()
    fifo=dict(re.findall(r"#pragma HLS STREAM variable=(\w+) depth=(\d+)",source))
    baseline_fifo=dict(re.findall(r"#pragma HLS STREAM variable=(\w+) depth=(\d+)",(BASELINE / "firmware" / f"{TOP}.cpp").read_text()))
    assert fifo == baseline_fifo, (fifo,baseline_fifo)
    write_json(out / "fifo_depths.json",fifo)
    # compile() rewrites project Tcl; preserve the backend defaults.
    print(f"RF{rf}: compiling and validating 10000 images",flush=True)
    hls_model.compile()
    started=time.time()
    prediction=np.asarray(hls_model.predict(x),dtype=np.float32).reshape(reference.shape)
    ref_class,hls_class=np.argmax(reference,axis=1),np.argmax(prediction,axis=1)
    ref_correct,hls_correct=int(np.sum(ref_class==labels)),int(np.sum(hls_class==labels))
    agreement=int(np.sum(ref_class==hls_class))
    error=np.abs(reference-prediction)
    confusion=np.zeros((10,10),dtype=np.int64)
    np.add.at(confusion,(labels,hls_class),1)
    status="passed" if agreement/len(labels)>=0.999 and hls_correct>=ref_correct-10 else "failed"
    validation={"status":status,"samples":len(labels),"keras_correct":ref_correct,
        "keras_accuracy_percent":100*ref_correct/len(labels),"hls_correct":hls_correct,
        "hls_accuracy_percent":100*hls_correct/len(labels),"hls_wilson_95":wilson(hls_correct,len(labels)),
        "accuracy_delta_hls_minus_keras_pp":100*(hls_correct-ref_correct)/len(labels),
        "keras_hls_argmax_agreement":agreement,"keras_hls_argmax_agreement_percent":100*agreement/len(labels),
        "logit_mean_absolute_error":float(error.mean()),"logit_max_absolute_error":float(error.max()),
        "confusion_matrix_rows_true_columns_predicted":confusion.tolist(),
        "acceptance":{"minimum_argmax_agreement_percent":99.9,"maximum_accuracy_loss_images":10},
        "cpp_evaluation_host_seconds":time.time()-started}
    baseline=np.load(BASELINE / "cpp_predictions_10000.npz")
    validation["baseline_hls_argmax_agreement"]=int(np.sum(hls_class==baseline["hls_predictions"]))
    validation["baseline_hls_logits_max_absolute_error"]=float(np.max(np.abs(prediction-baseline["hls_logits"])))
    np.savez_compressed(out / "cpp_predictions_10000.npz",keras_logits=reference,hls_logits=prediction,
        labels=labels,keras_predictions=ref_class,hls_predictions=hls_class)
    write_json(out / "cpp_validation.json",validation)
    manifest={"model":str(MODEL),"model_sha256":sha(MODEL),"data_sha256":sha(DATA),
        "global_reuse_factor":rf,"design":design,"hls_config":config,"project_name":TOP,
        "effective_reuse":{row["layer"]:row["effective_rf"] for row in resolved},
        "fifo_depths":fifo,"fifo_optimization":False,"rtl_samples":10,"rtl_indices":indices.tolist(),
        "software":{"python":platform.python_version(),"tensorflow":tf.__version__,"keras":keras.__version__,
        "hls4ml":hls4ml.__version__,"numpy":np.__version__},
        "testbench_changes":"10 fixed class-balanced images; default hls4ml simulation tracing; no hardware change",
        "reference_note":"design identifies the previous layer-wise baseline; effective H1 RF is recorded separately"}
    write_json(out / "configuration_manifest.json",manifest)
    print(json.dumps(validation,indent=2),flush=True)
    if status != "passed":
        raise RuntimeError("Numerical validation failed; HLS not started")
    if args.skip_build:
        return
    print(f"RF{rf}: C simulation, synthesis, RTL cosimulation and export",flush=True)
    report=hls_model.build(reset=True,csim=True,synth=True,cosim=True,validation=True,
        export=True,vsynth=False,fifo_opt=False,log_to_stdout=True)
    write_json(out / "build_report.json",report)
    assert report["CosimReport"]["Status"] == "Pass",report.get("CosimReport")
    csim=np.asarray(report["CSimResults"],dtype=float)
    cosim=np.asarray(report["CosimResults"],dtype=float)
    assert csim.shape == cosim.shape == (10,10)
    assert np.array_equal(csim,cosim)
    assert np.allclose(csim,prediction[indices],atol=1e-4,rtol=0)
    export_audit=audit_ip(out,TOP)
    export_audit.update(rtl_samples=10,cosim_matches_csim=True,csim_matches_cpp=True)
    write_json(out / "export_audit.json",export_audit)
    print(f"RF{rf}: IP export and validation passed",flush=True)

if __name__ == "__main__":
    main()
