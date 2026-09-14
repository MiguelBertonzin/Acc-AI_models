from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "mlp": {
        "name": "MLP Iris",
        "model": ROOT / "models/h5/iris_mlp_clean.h5",
        "test": ROOT / "data/mlp/iris_test.npz",
        "calibration": ROOT / "data/mlp/iris_calibration_train.npz",
        "input_key": "features",
        "labels_key": "labels",
        "shape": (1, 4),
        "normalize_uint8": False,
    },
    "lenet": {
        "name": "LeNet MNIST",
        "model": ROOT / "models/h5/lenet_mnist_final.h5",
        "test": ROOT / "data/lenet/mnist_test_uint8.npz",
        "calibration": ROOT / "data/lenet/mnist_calibration_train_uint8.npz",
        "input_key": "images",
        "labels_key": "labels",
        "shape": (1, 28, 28, 1),
        "normalize_uint8": True,
    },
    "resnet8": {
        "name": "ResNet8 CIFAR-10",
        "model": ROOT / "models/h5/resnet8_cifar10_keras3_no_softmax.h5",
        "test": ROOT / "data/resnet8/cifar10_test_uint8.npz",
        "calibration": ROOT / "data/resnet8/cifar10_calibration_train_uint8.npz",
        "input_key": "images",
        "labels_key": "labels",
        "shape": (1, 32, 32, 3),
        "normalize_uint8": True,
    },
}


def prepare_sample(spec, sample):
    import numpy as np

    value = np.asarray(sample, dtype=np.float32)
    if spec["normalize_uint8"]:
        value = value / np.float32(255.0)
    return value.reshape(spec["shape"])

