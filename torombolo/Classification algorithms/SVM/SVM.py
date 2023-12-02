import numpy as np

class SimpleSVM:

  def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100):
    self.alpha = learning_rate
    self.lambda_param = lambda_param
    self.n_iters = n_iters
    self.w = None
    self.b = None

  def fit(self, X, y):
    n_samples, n_features = X.shape

    # valores iniciales
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
    approx = np.dot(X, self.w) + self.b
    return np.sign(approx)
