#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from hls_common import DEFAULT_MODEL, HLS_ROOT, ORIGINAL_MODEL, final_activation, sha256


def describe(model, path):
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "parameters": model.count_params(),
        "layers": len(model.layers),
        "last_layer": model.layers[-1].name,
        "last_activation": final_activation(model),
    }


def main() -> None:
    import tensorflow as tf

    original = tf.keras.models.load_model(ORIGINAL_MODEL, compile=False)
    logits_model = tf.keras.models.load_model(DEFAULT_MODEL, compile=False)

    rng = np.random.default_rng(1234)
    x = rng.normal(size=(8, 32, 32, 3)).astype(np.float32)
    probabilities = np.asarray(original(x, training=False))
    logits = np.asarray(logits_model(x, training=False))
    reconstructed = tf.nn.softmax(logits, axis=-1).numpy()

    weight_errors = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(original.get_weights(), logits_model.get_weights(), strict=True)
    ]
    payload = {
        "original": describe(original, ORIGINAL_MODEL),
        "no_softmax": describe(logits_model, DEFAULT_MODEL),
        "same_parameter_count": original.count_params() == logits_model.count_params(),
        "max_weight_error": max(weight_errors, default=0.0),
        "max_softmax_reconstruction_error": float(
            np.max(np.abs(probabilities - reconstructed))
        ),
        "argmax_agreement": float(
            np.mean(np.argmax(probabilities, axis=1) == np.argmax(logits, axis=1))
        ),
    }
    payload["equivalent"] = bool(
        payload["same_parameter_count"]
        and payload["max_weight_error"] == 0.0
        and np.allclose(probabilities, reconstructed, rtol=1e-5, atol=1e-6)
        and payload["argmax_agreement"] == 1.0
    )
    out = HLS_ROOT / "reports/model_audit.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nRelatório: {out}")
    if not payload["equivalent"]:
        raise SystemExit("Os modelos não passaram na auditoria de equivalência.")


if __name__ == "__main__":
    main()

