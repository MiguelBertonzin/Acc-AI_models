"""Exporta o modelo LeNet no formato HDF5 e valida a equivalencia numerica."""

from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "lenet_mnist_final.keras"
DESTINATION = ROOT / "lenet_mnist_final.h5"


def main() -> None:
    model_keras = tf.keras.models.load_model(SOURCE, compile=False)
    model_keras.save(DESTINATION, include_optimizer=False)

    model_h5 = tf.keras.models.load_model(DESTINATION, compile=False)
    rng = np.random.default_rng(42)
    sample = rng.random((32, 28, 28, 1), dtype=np.float32)

    output_keras = model_keras(sample, training=False).numpy()
    output_h5 = model_h5(sample, training=False).numpy()
    max_abs_error = float(np.max(np.abs(output_keras - output_h5)))

    if not np.array_equal(output_keras, output_h5):
        raise RuntimeError(
            "A conversao nao preservou as saidas bit a bit; "
            f"erro absoluto maximo: {max_abs_error:.9g}"
        )

    print(f"Origem: {SOURCE}")
    print(f"Destino: {DESTINATION}")
    print(f"Entrada: {model_h5.input_shape}")
    print(f"Saida: {model_h5.output_shape}")
    print(f"Parametros: {model_h5.count_params()}")
    print(f"Erro absoluto maximo: {max_abs_error:.9g}")
    print("Validacao: saidas identicas bit a bit em 32 entradas deterministicas")


if __name__ == "__main__":
    main()
