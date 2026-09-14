# Modelos usados no fluxo

Os modelos-fonte permanecem um nível acima para não criar cópias divergentes:

- `../../resnet8_cifar10_keras3.h5`: modelo original, saída softmax;
- `../../resnet8_cifar10_keras3_no_softmax.h5`: modelo canônico para hardware, saída em logits.

Checksums SHA-256 verificados em 2026-08-26:

```text
9cc7cd3ea1c9603501f0c77dc0848b41d0bd136f80cc1d328881ed89cbb9cb1b  resnet8_cifar10_keras3.h5
047c190c70ea5901d2390af47b04f9d1e015c66cd7ff8bfc2a1e4f084154761e  resnet8_cifar10_keras3_no_softmax.h5
```

Ambos têm entrada `(32, 32, 3)`, saída `(10,)`, 78.714 parâmetros e os mesmos pesos. O script `01_inspect_models.py` refaz essa auditoria e grava o resultado em `reports/`.

