# therrshan-datasetloader

A lightweight Python package for loading classic machine learning datasets like **MNIST**, **Fashion MNIST**, and **20 Newsgroups** directly from local files — with support for normalization, pandas/numpy format switching, and split selection.

## Features

- **Fast local loading** - No internet required after initial setup
- **Multiple datasets** - MNIST, Fashion MNIST, 20 Newsgroups
- **Format flexibility** - Switch between pandas DataFrames and numpy arrays
- **Automatic normalization** - Built-in data preprocessing options. (Working on adding more)
- **Split selection** - Choose train, test, or validation sets
- **Lightweight** - Minimal dependencies for quick installation

## Installation

```bash
pip install therrshan-datasetloader
```

## Quick Start

```python
from datasetloader.mnist import MNISTLoader
from datasetloader.fashion_mnist import FashionMNISTLoader
from datasetloader.newsgroups import TwentyNewsLoader

# MNIST
mnist = MNISTLoader(normalize=True, as_dataframe=True)
X, y = mnist.load("train")
print("MNIST:", X.shape, y.shape)

# Fashion MNIST
fashion = FashionMNISTLoader(normalize=True, as_dataframe=False)
Xf, yf = fashion.load("test")
print("Fashion MNIST:", Xf.shape, yf.shape)

# 20NG
news = TwentyNewsLoader(as_dataframe=True)
Xn, yn = news.load("train")
print("20NG:", Xn.shape, yn.shape)
```

## Supported Datasets (More coming soon)

#### MNIST

#### Fashion MNIST

#### 20 Newsgroups


## Data Directory Structure

The package expects datasets to be stored locally in the following structure:

```
data/
├── mnist/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── fashion-mnist/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
└── 20newsgroups/
    ├── train/
    └── test/
```

## License

MIT License

---

**Note**: This package is designed for local development and assumes you have datasets downloaded locally.