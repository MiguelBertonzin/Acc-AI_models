# Dados do CIFAR-10

Execute:

```bash
python scripts/02_prepare_cifar10.py
```

Arquivos gerados apenas nesta pasta:

```text
cifar10_x_test.npy  # (10000, 32, 32, 3), float32, intervalo [0, 1]
cifar10_y_test.npy  # (10000,), uint8
dataset_manifest.json
```

Os `.npy` não devem ser versionados; são reproduzíveis a partir do CIFAR-10 distribuído pelo Keras.

