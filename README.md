# ee298z
Shared code for the EE 298Z - Deep Learning class

## Keras Usage

### Pre-generate corrupted copy (easier)
```python
from keras.datasets import mnist
from hw2.transforms import *

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_corrupted = corrupt_mnist_copy(x_train)

# Scale to [0, 1]
x_train /= 255
x_train_corrupted /= 255
```

### Corrupt images on-the-fly
```python
from keras.datasets import mnist
from hw2.transforms import *

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

## PyTorch Usage

```python
from torchvision import datasets, transforms
from hw2.transforms import *

data = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           CorruptMNIST(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
```
