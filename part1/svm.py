import numpy as np
from sklearn.svm import LinearSVC

### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classification

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    # Initialize the linear SVM model with the specified parameters
    svm_model = LinearSVC(random_state=0, C=0.1)

    # Train the model
    svm_model.fit(train_x, train_y)

    # Predict the labels for the test set
    pred_test_y = svm_model.predict(test_x)

    return pred_test_y

def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classification using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    # Initialize the linear SVM model with the specified parameters
    svm_model = LinearSVC(random_state=0, C=0.1)

    # Train the model on the entire training dataset
    svm_model.fit(train_x, train_y)

    # Predict the labels for the test set
    pred_test_y = svm_model.predict(test_x)

    return pred_test_y

def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

# Example usage with hypothetical data
# Replace these with your actual training and test data
train_x = np.random.rand(1000, 784)  # 1000 examples, 784 features (28x28 pixels)
train_y = np.random.randint(0, 10, 1000)  # Digits 0-9
test_x = np.random.rand(200, 784)  # 200 test examples
test_y = np.random.randint(0, 10, 200)  # Digits 0-9

# Train the multi-class SVM and compute test error
pred_test_y = multi_class_svm(train_x, train_y, test_x)
test_error = compute_test_error_svm(test_y, pred_test_y)
print(f"Test error: {test_error}")
