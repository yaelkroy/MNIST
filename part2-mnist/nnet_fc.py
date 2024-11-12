import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def get_data():
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def run_model(batch_size, lr, momentum, activation_function):
    np.random.seed(12321)
    torch.manual_seed(12321)

    X_train, y_train, X_dev, y_dev, X_test, y_test = get_data()

    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    if activation_function == 'ReLU':
        model = nn.Sequential(
                  nn.Linear(784, 128),
                  nn.ReLU(),
                  nn.Linear(128, 10),
                )
    elif activation_function == 'LeakyReLU':
        model = nn.Sequential(
                  nn.Linear(784, 128),
                  nn.LeakyReLU(),
                  nn.Linear(128, 10),
                )

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    # Evaluate the model on the validation data
    loss, accuracy = run_epoch(dev_batches, model.eval(), None)
    print(f"Validation accuracy for {activation_function}, batch_size={batch_size}, lr={lr}, momentum={momentum}: {accuracy}")

    return accuracy

# Perform grid search with hidden layer size of 128
# Baseline (no modifications)
baseline_acc = run_model(32, 0.1, 0, 'ReLU')

# Batch size 64
batch_size_acc = run_model(64, 0.1, 0, 'ReLU')

# Learning rate 0.01
learning_rate_acc = run_model(32, 0.01, 0, 'ReLU')

# Momentum 0.9
momentum_acc = run_model(32, 0.1, 0.9, 'ReLU')

# LeakyReLU activation
leaky_relu_acc = run_model(32, 0.1, 0, 'LeakyReLU')

# Print results
print(f"Baseline accuracy: {baseline_acc}")
print(f"Batch size 64 accuracy: {batch_size_acc}")
print(f"Learning rate 0.01 accuracy: {learning_rate_acc}")
print(f"Momentum 0.9 accuracy: {momentum_acc}")
print(f"LeakyReLU activation accuracy: {leaky_relu_acc}")
