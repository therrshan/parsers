"""
MNIST/Fashion MNIST Dataset Parser

This script parses either the MNIST or Fashion MNIST dataset from raw files.
Assumes the raw files are already downloaded and stored in their respective data folders.

Usage:
    from mnist_parser import parse_mnist
    
    # Get the full MNIST dataset with standard normalization as numpy arrays
    X, y = parse_mnist(dataset='mnist', normalize='standard', return_type='numpy')
    
    # Get Fashion MNIST train/test split with minmax normalization as numpy arrays
    X_train, X_test, y_train, y_test = parse_mnist(
        dataset='fashion',
        normalize='minmax', 
        split=True, 
        train_ratio=0.8, 
        return_type='numpy'
    )
    
    # Get the full MNIST dataset as a pandas DataFrame
    df = parse_mnist(dataset='mnist', normalize='minmax', return_type='dataframe')
    
    # Get Fashion MNIST train/test split as a single pandas DataFrame with a 'split' column
    df = parse_mnist(dataset='fashion', normalize='standard', split=True, return_type='dataframe')
"""

import os
from os.path import join
import numpy as np

FASHION_MNIST_LABELS = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

MNIST_LABELS = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9'
}


def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    if method == 'minmax':
        return data.astype(np.float32) / 255.0
    
    elif method == 'standard':
        data = data.astype(np.float32)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    
    elif method == 'robust':
        data = data.astype(np.float32)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        iqr = q75 - q25
        median = np.median(data, axis=0)
        return (data - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def format_output(data, return_type: str = 'numpy', dataset: str = 'mnist'):

    if return_type == 'numpy':
        return data
    
    elif return_type == 'dataframe':
        import pandas as pd
        
        label_map = MNIST_LABELS if dataset == 'mnist' else FASHION_MNIST_LABELS
        
        if len(data) == 2:  
            X, y = data
            if len(X.shape) > 2:

                X = X.reshape(X.shape[0], -1)
            df = pd.DataFrame(X)
            df['label'] = y

            df['label_name'] = df['label'].map(label_map)
            return df
        
        elif len(data) == 4:
            X_train, X_test, y_train, y_test = data
 
            if len(X_train.shape) > 2:
                X_train = X_train.reshape(X_train.shape[0], -1)
            if len(X_test.shape) > 2:
                X_test = X_test.reshape(X_test.shape[0], -1)
                
            train_df = pd.DataFrame(X_train)
            train_df['label'] = y_train
            train_df['label_name'] = train_df['label'].map(label_map)
            train_df['split'] = 'train'
            
            test_df = pd.DataFrame(X_test)
            test_df['label'] = y_test
            test_df['label_name'] = test_df['label'].map(label_map)
            test_df['split'] = 'test'
            
            return pd.concat([train_df, test_df], ignore_index=True)
    
    else:
        raise ValueError(f"Unknown return type: {return_type}")


def parse_mnist(
    dataset: str = 'mnist',
    data_dir: str = None,
    normalize: str = None,
    split: bool = False,
    train_ratio: float = 0.8,
    random_state: int = 42,
    return_type: str = 'numpy',
    flatten: bool = False
):

    if data_dir is None:
        if dataset.lower() in ['mnist', 'mnistdigit', 'digit', 'digits']:
            data_dir = 'data/MNIST'
            dataset = 'mnist'
        elif dataset.lower() in ['fashion', 'fashion_mnist', 'fashion-mnist', 'fashionmnist']:
            data_dir = 'data/FashionMNIST'
            dataset = 'fashion'
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Choose 'mnist' or 'fashion'.")

    train_image_path = join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = join(data_dir, 'train-labels.idx1-ubyte')
    test_image_path = join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = join(data_dir, 't10k-labels.idx1-ubyte')

    with open(train_image_path, 'rb') as file: 
        image_train = np.frombuffer(file.read(), dtype=np.uint8)
        image_train = image_train[16:].reshape(-1, 28, 28)
    with open(train_labels_path, 'rb') as file:
        labels_train = np.frombuffer(file.read(), dtype=np.uint8)
        labels_train = labels_train[8:]

    with open(test_image_path, 'rb') as file: 
        image_test = np.frombuffer(file.read(), dtype=np.uint8)
        image_test = image_test[16:].reshape(-1, 28, 28)
    with open(test_labels_path, 'rb') as file:
        labels_test = np.frombuffer(file.read(), dtype=np.uint8)
        labels_test = labels_test[8:]

    if not split:
        images = np.vstack((image_train, image_test))
        labels = np.hstack((labels_train, labels_test))

        if flatten:
            images = images.reshape(images.shape[0], -1)

        if normalize:
            images = normalize_data(images, normalize)
        
        return format_output((images, labels), return_type, dataset)

    else:
        if flatten:
            image_train = image_train.reshape(image_train.shape[0], -1)
            image_test = image_test.reshape(image_test.shape[0], -1)

        if normalize:
            image_train = normalize_data(image_train, normalize)
            image_test = normalize_data(image_test, normalize)
        
        if train_ratio != 0.857:  
            np.random.seed(random_state)
            
            images = np.vstack((image_train, image_test))
            labels = np.hstack((labels_train, labels_test))

            if flatten and len(images.shape) > 2:
                images = images.reshape(images.shape[0], -1)

            indices = np.random.permutation(len(images))
            images = images[indices]
            labels = labels[indices]

            split_idx = int(len(images) * train_ratio)
            image_train = images[:split_idx]
            image_test = images[split_idx:]
            labels_train = labels[:split_idx]
            labels_test = labels[split_idx:]
        
        return format_output((image_train, image_test, labels_train, labels_test), return_type, dataset)


if __name__ == '__main__':

    X, y = parse_mnist(dataset='mnist')
    print(f"MNIST Data shape: {X.shape}, Labels shape: {y.shape}")
    
    X, y = parse_mnist(dataset='fashion')
    print(f"Fashion MNIST Data shape: {X.shape}, Labels shape: {y.shape}")

    df = parse_mnist(dataset='mnist', normalize='standard', return_type='dataframe')
    print(f"MNIST DataFrame shape: {df.shape}")
    print(f"MNIST DataFrame columns: {df.columns}")

    df = parse_mnist(dataset='fashion', normalize='standard', return_type='dataframe')
    print(f"Fashion MNIST DataFrame shape: {df.shape}")
    print(f"Fashion MNIST DataFrame columns: {df.columns}")