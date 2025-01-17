import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each data point
        lambda_factor - the regularization constant (scalar)
    
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
                represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # Compute the transpose of X
    X_transpose = np.transpose(X)
    
    # Compute X^T * X
    X_transpose_X = np.dot(X_transpose, X)
    
    # Add the regularization term lambda * I
    identity_matrix = np.identity(X.shape[1])
    regularization_term = lambda_factor * identity_matrix
    
    # Compute the inverse of (X^T * X + lambda * I)
    inverse_term = np.linalg.inv(X_transpose_X + regularization_term)
    
    # Compute the final term (X^T * Y)
    X_transpose_Y = np.dot(X_transpose, Y)
    
    # Compute the closed form solution theta
    theta = np.dot(inverse_term, X_transpose_Y)
    
    return theta


### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
