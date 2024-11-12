import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from train_utils import batchify_data, run_epoch, Flatten
import utils_multiMNIST as U

path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 50  # Increased the number of epochs
num_classes = 10
img_rows, img_cols = 42, 28  # input image dimensions

class CNN(nn.Module):
    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, (3, 3), padding=1)  # Added another convolutional layer
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d((2, 2))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(256 * 2 * 1, 512)  # Adjusted the size according to the pooling and convolution operations
        self.dropout = nn.Dropout(0.5)
        self.fc2_1 = nn.Linear(512, 10)  # Output layer for the first digit
        self.fc2_2 = nn.Linear(512, 10)  # Output layer for the second digit

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Added the fourth convolutional layer
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out_first_digit = self.fc2_1(x)
        out_second_digit = self.fc2_2(x)
        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension)

    # Training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Learning rate scheduler
    n_epochs = nb_epoch  # Use the updated number of epochs

    # Train
    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_batches, model.train(), optimizer)
        print('Train | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_batches, model.eval(), None)
        print('Valid | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(val_loss[0], val_acc[0], val_loss[1], val_acc[1]))

        # Step the scheduler
        scheduler.step()

        # Save model
        torch.save(model, 'mnist_model_cnn.pt')

    # Evaluate the model on test data
    test_loss, test_acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}  accuracy2: {:.6f}'.format(test_loss[0], test_acc[0], test_loss[1], test_acc[1]))
    print('Final Test Accuracy: accuracy1: {:.6f}, accuracy2: {:.6f}'.format(test_acc[0], test_acc[1]))
    print('Overall Test Accuracy: {:.6f}'.format((test_acc[0] + test_acc[1]) / 2))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edX
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
