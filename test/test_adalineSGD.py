import sys
import os

# Obtén la ruta del directorio padre (el directorio raíz de tu proyecto)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from torombolo.perceptron import AdalineSGD
import numpy as np
import unittest

class TestAdalineSGD(unittest.TestCase):
    def test_fit(self):
        X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_train = np.array([0, 0, 1, 1])

        adaline = AdalineSGD()
        adaline.fit(X_train, y_train)

        self.assertEqual(adaline.predict(np.array([0, 0])), 0)
        self.assertEqual(adaline.predict(np.array([5, 5])), 1)

    def test_partial_fit(self):
        X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_train = np.array([0, 1, 0, 1])

        adaline = AdalineSGD()
        adaline.partial_fit(X_train, y_train)

        self.assertEqual(adaline.predict(np.array([0, 0])), 0)
        self.assertEqual(adaline.predict(np.array([5, 5])), 1)

    def test_predict(self):
        X_test = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_test = np.array([0, 0, 1, 1])

        adaline = AdalineSGD()
        adaline.fit(X_test, y_test)

        self.assertEqual(adaline.predict(np.array([0, 0])), 0)
        self.assertEqual(adaline.predict(np.array([5, 5])), 1)

if __name__ == '__main__':
    unittest.main()