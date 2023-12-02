# Author: Erik García (Erik172)
import numpy as np

class LinearRegression:
    """
    Linear regression model.

    Parameters:
    - learning_rate (float): The learning rate for gradient descent. Default is 0.001.
    - n_iters (int): The number of iterations for gradient descent. Default is 100.

    Attributes:
    - alpha (float): The learning rate for gradient descent.
    - n_iters (int): The number of iterations for gradient descent.
    - slope_ (float): The slope of the linear regression line.
    - intercept_ (float): The y-intercept of the linear regression line.
    """

    def __init__(self, learning_rate=0.001, n_iters=100):
        self.alpha = learning_rate
        self.n_iters = n_iters
        self.slope_ = 0
        self.intercept_ = 0
        self.costs = []

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target values.

        Returns:
        - self (LinearRegression): The fitted LinearRegression object.
        """
        self.slope_, self.intercept_ = self._gradient_descent(X, y)
        return self

    def _get_gradient_m(self, X, y, m, b):
        """
        Calculate the gradient of the slope (m) parameter.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target values.
        - m (float): The current value of the slope parameter.
        - b (float): The current value of the y-intercept parameter.

        Returns:
        - slope_gradient (float): The gradient of the slope parameter.
        """
        diff = 0

        for i in range(len(X)):
            diff += X[i] * (y[i] - (m * X[i] + b))

        slope_gradient = -2/len(X) * diff
        return slope_gradient

    def _get_gradient_b(self, X, y, m, b):
        """
        Calculate the gradient of the y-intercept (b) parameter.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target values.
        - m (float): The current value of the slope parameter.
        - b (float): The current value of the y-intercept parameter.

        Returns:
        - b_gradient (float): The gradient of the y-intercept parameter.
        """
        diff = 0

        for i in range(len(X)):
            diff += (y[i] - (m * X[i] + b))

        b_gradient = -2/len(X) * diff
        return b_gradient
    
    def _step_gradient(self, X, y, slope_current, b_current):
        """
        Perform a single step of gradient descent.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target values.
        - slope_current (float): The current value of the slope parameter.
        - b_current (float): The current value of the y-intercept parameter.

        Returns:
        - slope_updated (float): The updated value of the slope parameter.
        - b_updated (float): The updated value of the y-intercept parameter.
        """
        slope_gradient = self._get_gradient_m(X, y, slope_current, b_current)
        b_gradient = self._get_gradient_b(X, y, slope_current, b_current)

        slope_updated = slope_current - self.alpha * slope_gradient
        b_updated = b_current - self.alpha * b_gradient

        return slope_updated, b_updated
    
    def _gradient_descent(self, X, y):
        """
        Perform gradient descent to find the optimal values of the slope and y-intercept parameters.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target values.

        Returns:
        - slope_optimal (float): The optimal value of the slope parameter.
        - b_optimal (float): The optimal value of the y-intercept parameter.
        """
        for _ in range(self.n_iters):
            self.slope_, self.intercept_ = self._step_gradient(X, y,self.slope_, self.intercept_)
            self.costs.append(self.score(X,y))

            # if _ % 20 == 0:
            #     print(f"Cost at iteration {_}: {self.costs[_]}")
            # if _ > 0 and self.costs[_] > self.costs[_-1]:
            #     break
            
        return self.slope_, self.intercept_
    
    def predict(self, X):
        """
        Predict the target values for the given input features.

        Parameters:
        - X (array-like): The input features.

        Returns:
        - y_pred (array-like): The predicted target values.
        """
        y_pred =self.slope_ * X + self.intercept_
        return y_pred
    
    def score(self, X, y):
        """
        Calculate the mean squared error of the model predictions.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target values.

        Returns:
        - score (float): The mean squared error.
        """
        y_pred = self.predict(X)
        score = np.sum((y_pred - y)**2) / len(y)
        return score
    
    def __repr__(self):
        return f"LinearRegression(learning_rate={self.alpha}, n_iters={self.n_iters})"
    
    def __str__(self):
        return f"LinearRegression(learning_rate={self.alpha}, n_iters={self.n_iters})"