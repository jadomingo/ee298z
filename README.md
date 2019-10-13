# ee298z
Shared code for the EE 298Z - Deep Learning class

## Homework 2: Image Restoration using Autoencoders

### Additional Python Packages Required
- ```scikit-image``` >= 0.15.0

### Dataset Generation for Training

#### Keras

##### Generate corrupted copy (easier)
```python
from keras.datasets import mnist
from hw2.transforms import corrupt_mnist_copy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_corrupted = corrupt_mnist_copy(x_train)

# Scale to [0, 1]
x_train /= 255
x_train_corrupted /= 255
```

##### Corrupt images on-the-fly
```python
from keras.datasets import mnist
from hw2.transforms import corrupt_mnist_generator

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Scale to [0, 1]
x_train /= 255

x_train_corrupted_gen = corrupt_mnist_generator(x_train, value=1.)

def data_gen():
    orig = x_train[i]
    corrupted = next(x_train_corrupted_gen)
    yield orig, corrupted

train_gen = data_gen()
...
model.fit_generator(train_gen, ...)
```

#### PyTorch
Note: data is generated on-the-fly
```python
from torchvision import datasets, transforms
from hw2.transforms import CorruptMNIST

data = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            CorruptMNIST()
                        ])),
```

### Benchmark Code for Testing

#### Metrics
| Metric | Min | Max | Baseline | Description |
| ------ | --- | --- | -------- | ----------- |
| Classifier Score | `0` | `100` | `78.49` (PyTorch) / `77.33` (Keras) | Measures quality of inpainting |
| SSIM Score | `0` | `100` | `50` | Measures capability to detect and preserve uncorrupted pixels |

The baseline is the number to beat.

#### Keras
```python
from keras.datasets import mnist
from hw2.benchmark_keras import test_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale to [0, 1]
x_test = x_test / 255

# model is your Keras model
# DO NOT convert y_test, i.e. don't use keras.utils.to_categorical()
test_model(model, x_test, y_test, batch_size=100)
```

#### PyTorch
```python
from torchvision import datasets, transforms
from hw2.benchmark_torch import test_model

data = datasets.MNIST('../data', train=False, download=True,
                        transform=transforms.ToTensor()),

# model is your PyTorch model
test_model(model, data, batch_size=100)
```

#### Sample Benchmark Output
```
Classifier score: 78.49
SSIM score: 100.00
```
