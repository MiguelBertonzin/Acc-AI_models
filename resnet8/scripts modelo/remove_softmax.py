#!/usr/bin/env python3
"""Create an equivalent Keras model whose final Dense returns logits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the final Dense softmax activation with a linear activation."
    )
    parser.add_argument("input", type=Path, help="Input Keras .h5 model")
    parser.add_argument("output", type=Path, help="Output Keras .h5 model")
    return parser.parse_args()


def build_logits_model(model: keras.Model) -> keras.Model:
    if len(model.outputs) != 1:
        raise ValueError("Expected a model with exactly one output.")

    output_layer = model.layers[-1]
    if not isinstance(output_layer, keras.layers.Dense):
        raise TypeError(
            f"Expected the output layer to be Dense, got {type(output_layer).__name__}."
        )
    if keras.activations.serialize(output_layer.activation) != "softmax":
        raise ValueError(
            f"Expected softmax in layer {output_layer.name!r}, got "
            f"{keras.activations.serialize(output_layer.activation)!r}."
        )

    def clone_layer(layer: keras.layers.Layer) -> keras.layers.Layer:
        config = layer.get_config()
        if layer.name == output_layer.name:
            config["activation"] = "linear"
        return layer.__class__.from_config(config)

    logits_model = keras.models.clone_model(model, clone_function=clone_layer)
    logits_model.set_weights(model.get_weights())
    return logits_model


def verify_equivalence(
    softmax_model: keras.Model, logits_model: keras.Model, seed: int = 1234
) -> float:
    sample_shape = []
    for dim in softmax_model.input_shape[1:]:
        if dim is None:
            raise ValueError("Cannot validate a model with a dynamic non-batch input shape.")
        sample_shape.append(dim)

    rng = np.random.default_rng(seed)
    sample = rng.normal(size=(4, *sample_shape)).astype(np.float32)
    probabilities = softmax_model(sample, training=False).numpy()
    logits = logits_model(sample, training=False).numpy()
    probabilities_from_logits = tf.nn.softmax(logits, axis=-1).numpy()

    max_error = float(np.max(np.abs(probabilities - probabilities_from_logits)))
    if not np.allclose(probabilities, probabilities_from_logits, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"Softmax equivalence check failed (max error: {max_error}).")
    if not np.array_equal(
        np.argmax(probabilities, axis=-1), np.argmax(logits, axis=-1)
    ):
        raise RuntimeError("Class-index equivalence check failed.")
    return max_error


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different.")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    source_model = keras.models.load_model(args.input, compile=False)
    logits_model = build_logits_model(source_model)
    max_error = verify_equivalence(source_model, logits_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logits_model.save(args.output)

    print(f"Saved: {args.output}")
    print(f"Output activation: {keras.activations.serialize(logits_model.layers[-1].activation)}")
    print(f"Output shape: {logits_model.output_shape}")
    print(f"Maximum softmax reconstruction error: {max_error:.3e}")
    print("Final class index: np.argmax(model_output, axis=-1)")


if __name__ == "__main__":
    main()
