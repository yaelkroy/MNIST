import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import scipy.sparse as sparse

sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

# TODO: first fill out functions in linear_regression.py, otherwise the functions below will not work


def run_linear_regression_on_MNIST(lambda_factor=1):
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))


#######################################################################
# 3. Support Vector Machine
#######################################################################

# TODO: first fill out functions in svm.py, or the functions below will not work

def run_svm_one_vs_rest_on_MNIST():
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################

# TODO: first fill out functions in softmax.py, or run_softmax_on_MNIST will not work

def run_softmax_on_MNIST(temp_parameter=1):
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1e-4, num_iterations=150, k=10)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    write_pickle_data(theta, "./theta.pkl.gz")
    return test_error


print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))

#######################################################################
# 6. Changing Labels
#######################################################################

def update_y_mod3(Y):
    return Y % 3

def run_softmax_on_MNIST_mod3(temp_parameter=1):
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y_mod3 = update_y_mod3(train_y)
    test_y_mod3 = update_y_mod3(test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y_mod3, temp_parameter, alpha=0.3, lambda_factor=1e-4, num_iterations=150, k=3)
    test_error = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)
    return theta, test_error

print('Softmax (mod 3) test_error:', run_softmax_on_MNIST_mod3(temp_parameter=1))

# TODO: Run run_softmax_on_MNIST_mod3(), report the error rate


#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.

n_components = 18

train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.

# TODO: Train your softmax regression model using (train_pca, train_y)
#       and evaluate its accuracy on (test_pca, test_y).

theta_pca, cost_function_history_pca = softmax_regression(train_pca, train_y, temp_parameter, alpha=0.3, lambda_factor=1e-4, num_iterations=150, k=10)
test_error_pca = compute_test_error(test_pca, test_y, theta_pca, temp_parameter)
print('Softmax regression on PCA-transformed data test_error:', test_error_pca)

# TODO: Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
plot_PC(train_x[range(0, 100), :], pcs, train_y[range(0, 100)], feature_means)

# TODO: Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
firstimage_reconstructed = reconstruct_PC(train_pca[0, :], pcs, n_components, feature_means)
plot_images(firstimage_reconstructed)
plot_images(train_x[0, :])

secondimage_reconstructed = reconstruct_PC(train_pca[1, :], pcs, n_components, feature_means)
plot_images(secondimage_reconstructed)
plot_images(train_x[1, :])


## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set

n_components_10 = 10
train_pca10 = project_onto_PC(train_x, pcs, n_components_10, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components_10, feature_means)

# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)

# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.

# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).

theta_cube, cost_function_history_cube = softmax_regression(train_cube, train_y, temp_parameter, alpha=0.3, lambda_factor=1e-4, num_iterations=150, k=10)
test_error_cube = compute_test_error(test_cube, test_y, theta_cube, temp_parameter)
print('Softmax regression on cubic kernel-transformed data test_error:', test_error_cube)

#######################################################################
# 8. Effects of Adjusting Temperature
#######################################################################

def run_softmax_with_different_temperatures():
    temperatures = [0.5, 1.0, 2.0]
    for temp in temperatures:
        print(f'Running softmax with temperature {temp}')
        test_error = run_softmax_on_MNIST(temp_parameter=temp)
        print(f'Test error for temperature {temp}: {test_error}')

run_softmax_with_different_temperatures()
def run_softmax_with_different_temperatures():
    temperatures = [0.5, 1.0, 2.0]
    for temp in temperatures:
        print(f'Running softmax with temperature {temp}')
        test_error = run_softmax_on_MNIST(temp_parameter=temp)
        print(f'Test error for temperature {temp}: {test_error}')

run_softmax_with_different_temperatures()


def run_softmax_on_MNIST(temp_parameter=1):
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1e-4, num_iterations=150, k=10)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    write_pickle_data(theta, "./theta.pkl.gz")
    
    # Updating the labels to mod 3
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    
    # Compute the error rate for the new labels (mod 3)
    test_error_mod3 = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)
    
    # Print the error rates
    print('softmax test_error=', test_error)
    print('Error rate for labels mod 3:', test_error_mod3)
    
    return test_error, test_error_mod3

# Run the function
test_error, test_error_mod3 = run_softmax_on_MNIST(temp_parameter=1)


def run_softmax_with_different_temperatures():
    temperatures = [0.5, 1.0, 2.0]
    for temp in temperatures:
        print(f'Running softmax with temperature {temp}')
        test_error = run_softmax_on_MNIST(temp_parameter=temp)
        print(f'Test error for temperature {temp}: {test_error}')

run_softmax_with_different_temperatures()
n_components = 10

# Center the data
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)

# Project the data onto the first 10 principal components
train_pca10 = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca10 = project_onto_PC(test_x, pcs, n_components, feature_means)

## RBF Kernel SVM ##

# Train the SVM model using the RBF kernel on the PCA-transformed data
svm_model = SVC(kernel='rbf', random_state=0)
svm_model.fit(train_pca10, train_y)

# Predict the labels for the test set
pred_test_y = svm_model.predict(test_pca10)

# Compute the test error
test_error_svm = 1 - np.mean(pred_test_y == test_y)

# Print the test error
print('RBF SVM on 10-dimensional PCA data test_error:', test_error_svm)