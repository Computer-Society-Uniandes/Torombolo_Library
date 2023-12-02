import sys
import os

# Obtén la ruta del directorio padre (el directorio raíz de tu proyecto)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
from torombolo.perceptron import Perceptron
import unittest
import numpy as np
# from torombolo.perceptron.perceptron import Perceptron

class TestPerceptron(unittest.TestCase):
    def test_linearly_separable_data(self):
        X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_train = np.array([0, 0, 1, 1])

        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)

        self.assertEqual(perceptron.predict(np.array([0, 0])), 0)
        self.assertEqual(perceptron.predict(np.array([5, 5])), 1)

    def test_non_linearly_separable_data(self):
        X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_train = np.array([0, 1, 0, 1])

        perceptron = Perceptron()
        perceptron.fit(X_train, y_train)

        self.assertEqual(perceptron.predict(np.array([0, 0])), 0)
        self.assertEqual(perceptron.predict(np.array([5, 5])), 1)

    def test_accuracy(self):
        X_test = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_test = np.array([0, 0, 1, 1])

        perceptron = Perceptron()
        perceptron.fit(X_test, y_test)

        accuracy = perceptron.score(X_test, y_test)
        self.assertEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()