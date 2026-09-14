import os
import random
import numpy as np
import tensorflow as tf

# ============================================================
# Configuração para reprodutibilidade
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("LeNet - MNIST")
print("=" * 70)

print(f"TensorFlow: {tf.__version__}")

# ============================================================
# 1. Carregamento do conjunto MNIST
# ============================================================

print("\nCarregando MNIST...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("\nFormato original:")
print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"x_test : {x_test.shape}")
print(f"y_test : {y_test.shape}")

# ============================================================
# 2. Pré-processamento
# ============================================================

# Converter pixels de [0, 255] para [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Adicionar dimensão do canal:
# (28, 28) -> (28, 28, 1)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nFormato após pré-processamento:")
print(f"x_train: {x_train.shape}")
print(f"x_test : {x_test.shape}")

# ============================================================
# 3. Construção da arquitetura LeNet
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(28, 28, 1),
            name="input"
        ),

        # ----------------------------------------------------
        # Conv1
        # 28x28x1 -> 24x24x6
        # ----------------------------------------------------
        tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            name="conv1"
        ),

        # ----------------------------------------------------
        # Pool1
        # 24x24x6 -> 12x12x6
        # ----------------------------------------------------
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="pool1"
        ),

        # ----------------------------------------------------
        # Conv2
        # 12x12x6 -> 8x8x16
        # ----------------------------------------------------
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            name="conv2"
        ),

        # ----------------------------------------------------
        # Pool2
        # 8x8x16 -> 4x4x16
        # ----------------------------------------------------
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            name="pool2"
        ),

        # ----------------------------------------------------
        # Flatten
        # 4x4x16 -> 256
        # ----------------------------------------------------
        tf.keras.layers.Flatten(
            name="flatten"
        ),

        # ----------------------------------------------------
        # Dense 1
        # 256 -> 120
        # ----------------------------------------------------
        tf.keras.layers.Dense(
            120,
            activation="relu",
            name="dense1"
        ),

        # ----------------------------------------------------
        # Dense 2
        # 120 -> 84
        # ----------------------------------------------------
        tf.keras.layers.Dense(
            84,
            activation="relu",
            name="dense2"
        ),

        # ----------------------------------------------------
        # Saída
        # 84 -> 10
        #
        # Sem softmax: saída LINEAR, exatamente como descrito
        # na tabela do trabalho.
        # ----------------------------------------------------
        tf.keras.layers.Dense(
            10,
            activation=None,
            name="output"
        )
    ],
    name="LeNet_MNIST"
)

# ============================================================
# 4. Exibir arquitetura
# ============================================================

print("\n")
print("=" * 70)
print("ARQUITETURA")
print("=" * 70)

model.summary()

print("\nNúmero total de parâmetros:")
print(f"{model.count_params():,}".replace(",", "."))

# ============================================================
# 5. Verificação manual dos MACs
# ============================================================

# Conv1:
# 24 * 24 * 6 saídas
# cada saída = 5 * 5 * 1 MACs

conv1_macs = 24 * 24 * 6 * 5 * 5 * 1

# Conv2:
# 8 * 8 * 16 saídas
# cada saída = 5 * 5 * 6 MACs

conv2_macs = 8 * 8 * 16 * 5 * 5 * 6

# Dense1
dense1_macs = 256 * 120

# Dense2
dense2_macs = 120 * 84

# Saída
output_macs = 84 * 10

total_macs = (
    conv1_macs
    + conv2_macs
    + dense1_macs
    + dense2_macs
    + output_macs
)

print("\n")
print("=" * 70)
print("MACs POR INFERÊNCIA")
print("=" * 70)

print(f"Conv1  : {conv1_macs:,}")
print(f"Conv2  : {conv2_macs:,}")
print(f"Dense1 : {dense1_macs:,}")
print(f"Dense2 : {dense2_macs:,}")
print(f"Output : {output_macs:,}")
print("-" * 30)
print(f"TOTAL  : {total_macs:,}")

# ============================================================
# 6. Compilação
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    # A saída da rede é LINEAR.
    # Portanto, usamos from_logits=True.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),

    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)

# ============================================================
# 7. Callbacks
# ============================================================

callbacks = [

    # Salva automaticamente o modelo com melhor
    # acurácia de validação.
    tf.keras.callbacks.ModelCheckpoint(
        filepath="lenet_mnist_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    # Interrompe treinamento caso não exista melhora.
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]

# ============================================================
# 8. Treinamento
# ============================================================

print("\n")
print("=" * 70)
print("INICIANDO TREINAMENTO")
print("=" * 70)

history = model.fit(
    x_train,
    y_train,

    batch_size=128,

    epochs=30,

    validation_split=0.1,

    shuffle=True,

    callbacks=callbacks,

    verbose=1
)

# ============================================================
# 9. Avaliação no conjunto de teste
# ============================================================

print("\n")
print("=" * 70)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    batch_size=128,
    verbose=1
)

print("\nResultados finais:")
print(f"Loss teste     : {test_loss:.6f}")
print(f"Acurácia teste : {test_accuracy * 100:.4f}%")

# ============================================================
# 10. Salvar modelo
# ============================================================

print("\nSalvando modelos...")

# Formato moderno Keras
model.save(
    "lenet_mnist_final.keras"
)

# Formato HDF5 - útil posteriormente no hls4ml
model.save(
    "lenet_mnist_final.h5",
    include_optimizer=False
)

# Salvar apenas pesos, caso necessário
model.save_weights(
    "lenet_mnist_weights.h5"
)

print("\nArquivos gerados:")
print("  lenet_mnist_best.keras")
print("  lenet_mnist_final.keras")
print("  lenet_mnist_final.h5")
print("  lenet_mnist_weights.h5")

print("\nTreinamento concluído.")
