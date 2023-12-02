import numpy as np

class SimpleSVM:
  """
  A simple implementation of Support Vector Machine (SVM) classifier.

  Parameters:
  - learning_rate (float): The learning rate for gradient descent. Default is 0.001.
  - lambda_param (float): The regularization parameter. Default is 0.01.
  - n_iters (int): The number of iterations for training. Default is 100.

  Attributes:
  - alpha (float): The learning rate for gradient descent.
  - lambda_param (float): The regularization parameter.
  - n_iters (int): The number of iterations for training.
  - w (ndarray): The weight vector.
  - b (float): The bias term.

  Methods:
  - fit(X, y): Fit the SVM model to the training data.
  - predict(X): Predict the class labels for the input data.

  """

  def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100):
    self.alpha = learning_rate
    self.lambda_param = lambda_param
    self.n_iters = n_iters
    self.w = None
    self.b = None

  def fit(self, X, y):
    """
    Fit the SVM model to the training data.

    Parameters:
    - X (ndarray): The input features of shape (n_samples, n_features).
    - y (ndarray): The target labels of shape (n_samples,).

    """
    n_samples, n_features = X.shape

    # Initialize weights and bias
    self.w = np.zeros(n_features)
    self.b = 0

    for _ in range(self.n_iters):
      for i, x_i in enumerate(X):
        condition = y[i] * (np.dot(x_i, self.w) + self.b) >= 1
        if condition:
          self.w -= self.alpha * (2 * self.lambda_param * self.w)
          self.b -= self.b
        else:
          self.w -= self.alpha * ((2 * self.lambda_param * self.w) - np.dot(x_i, y[i]))
          self.b -= self.alpha * y[i]

  def predict(self, X):
    """
    Predict the class labels for the input data.

    Parameters:
    - X (ndarray): The input features of shape (n_samples, n_features).

    Returns:
    - ndarray: The predicted class labels of shape (n_samples,).

    """
    approx = np.dot(X, self.w) + self.b
    return np.sign(approx)
