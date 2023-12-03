import sys
import os

# Obtén la ruta del directorio padre (el directorio raíz de tu proyecto)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
from torombolo.linear_regression import LinearRegression
import unittest

class TestLinearRegression(unittest.TestCase):
    def test_fit(self):
        X_train = np.array([1, 2, 3, 4])
        y_train = np.array([2, 4, 6, 8])

        model = LinearRegression(learning_rate=0.01, n_iters=1000)
        model.fit(X_train, y_train)

        self.assertEqual(int(model.slope_), 1.0)
        self.assertEqual(int(model.intercept_), 0.0)

    def test_predict(self):
        X_train = np.array([1, 2, 3, 4])
        y_train = np.array([2, 4, 6, 8])

        model = LinearRegression(learning_rate=0.01, n_iters=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(np.array([5, 6, 7, 8]))
        y_pred = y_pred.astype(int)
        np.testing.assert_array_almost_equal(y_pred, np.array([9, 11, 13, 15]))

    def test_score(self):
        X_train = np.array([1, 2, 3, 4])
        y_train = np.array([2, 4, 6, 8])

        model = LinearRegression(learning_rate=0.01, n_iters=1000)
        model.fit(X_train, y_train)

        score = model.score(X_train, y_train)
        self.assertEqual(round(score, 1), 0.0)

    def test_accuracy_mse(self):
        X_train = np.array([1, 2, 3, 4])
        y_train = np.array([2, 4, 6, 8])

        model = LinearRegression(learning_rate=0.01, n_iters=1000)
        model.fit(X_train, y_train)

        accuracy = model.accuracy(X_train, y_train, metric="mse")
        self.assertEqual(round(accuracy, 1), 0.0)

    def test_accuracy_r2(self):
        X_train = np.array([1, 2, 3, 4])
        y_train = np.array([2, 4, 6, 8])

        model = LinearRegression(learning_rate=0.01, n_iters=1000)
        model.fit(X_train, y_train)

        accuracy = model.accuracy(X_train, y_train, metric="r2")
        self.assertEqual(round(accuracy, 2), 1.0)

if __name__ == '__main__':
    unittest.main()