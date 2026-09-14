# Decisão: usar o modelo `no_softmax`

O modelo canônico do acelerador será `resnet8_cifar10_keras3_no_softmax.h5`.

Para classificação top-1, a softmax não altera a classe escolhida, pois é estritamente crescente em cada logit sob o mesmo denominador:

```text
argmax(softmax(z)) = argmax(z)
```

Removê-la antes da conversão é preferível porque:

1. o grafo de entrada já representa exatamente o hardware desejado;
2. não há LUTs de exponencial/recíproco nem precisão específica de softmax;
3. não dependemos de uma diretiva `Skip=True` gerada pelo parser;
4. C simulation, RTL co-simulation e a interface AXI retornam todos a mesma entidade: 10 logits;
5. a rastreabilidade é mais clara: o modelo implantado é o arquivo cujo último `Dense` é linear.

O `.h5` original continua sendo a referência científica. O fluxo aceita esse arquivo e, nesse caso, exige encontrar exatamente a camada sintética `dense_softmax`, marca-a com `Skip=True` e usa os logits anteriores para o testbench. Essa rota serve para comparação, não como baseline principal.

Quando probabilidades calibradas forem realmente necessárias, a softmax pode ser executada no PS/CPU depois da leitura dos dez logits. Para top-1, basta `argmax`.

