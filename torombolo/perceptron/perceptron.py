# Autor: Erik García (Erik172)
import numpy as np

class Perceptron:
    """
    Un clasificador Perceptron.

    Parámetros:
    - learning_rate (float): La tasa de aprendizaje para el algoritmo Perceptron. El valor predeterminado es 0.01.
    - n_iter (int): El número de iteraciones para entrenar el Perceptron. El valor predeterminado es 100.

    Atributos:
    - w_ (ndarray): Los pesos del Perceptron después de ajustar los datos de entrenamiento.
    - b_ (ndarray): El término de sesgo del Perceptron después de ajustar los datos de entrenamiento.
    - errors_ (list): El número de clasificaciones incorrectas en cada iteración durante el entrenamiento.

    Métodos:
    - fit(X, y): Ajusta el Perceptron a los datos de entrenamiento.
    - activation(X): Calcula la función de activación para la entrada dada.
    - predict(X): Predice las etiquetas de clase para la entrada dada.
    - brute_predict(X): Predice los valores de salida sin procesar para la entrada dada.
    - score(X, y): Calcula la precisión del Perceptron en la entrada y etiquetas dadas.
    - accuracy(X, y, metric='accuracy'): Calcula la precisión, precisión, recuperación o puntuación F1 del Perceptron.
    - __repr__(): Devuelve una representación de cadena del Perceptron.
    - __str__(): Devuelve una representación de cadena del Perceptron.
    """

    def __init__(self, learning_rate=0.01, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
            """
            Fit the perceptron model to the training data.

            Parameters:
            X (array-like): The input samples.
            y (array-like): The target values.

            Returns:
            self (Perceptron): The fitted perceptron model.
            """
            self.w_ = np.random.rand(X.shape[1])
            self.b_ = np.random.rand(1)
            self.errors_ = []

            for _ in range(self.n_iter):
                error = 0
                for xi, yi in zip(X, y):
                    update = self.learning_rate * (yi - self.predict(xi))
                    self.w_ += update * xi
                    self.b_ += update
                    error += int(update != 0.0)
                self.errors_.append(error)
            return self

    def activation(self, X):
        return np.where(X >= 0.0, 1, 0)
    
    def predict(self, X):
        """
        Predicts the output for the given input data.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            array-like: Predicted output of shape (n_samples,).
        """
        weighted_sum = np.dot(X, self.w_) + self.b_
        return self.activation(weighted_sum)
    
    def brute_predict(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def score(self, X, y):
        """
        Calculates the accuracy score of the perceptron model on the given input data.

        Parameters:
        - X: Input features.
        - y: Target labels.

        Returns:
        - The accuracy score of the model.
        """
        return np.mean(self.predict(X) == y)
    
    def accuracy(self, X, y, metric='accuracy'):
        """
        Calculate the accuracy, precision, recall, or F1 score of the perceptron model.

        Parameters:
        - X (array-like): Input features.
        - y (array-like): Target labels.
        - metric (str): The metric to calculate. Supported metrics: accuracy, precision, recall, f1. Default is 'accuracy'.

        Returns:
        - float: The calculated metric value.

        Raises:
        - ValueError: If the specified metric is not supported.
        """
        if metric == 'accuracy':
            return np.mean(self.predict(X) == y)
        elif metric == 'precision':
            tp = np.sum(np.logical_and(self.predict(X) == 1, y == 1))
            fp = np.sum(np.logical_and(self.predict(X) == 1, y == 0))
            return tp / (tp + fp)
        elif metric == 'recall':
            tp = np.sum(np.logical_and(self.predict(X) == 1, y == 1))
            fn = np.sum(np.logical_and(self.predict(X) == 0, y == 1))
            return tp / (tp + fn)
        elif metric == 'f1':
            tp = np.sum(np.logical_and(self.predict(X) == 1, y == 1))
            fp = np.sum(np.logical_and(self.predict(X) == 1, y == 0))
            fn = np.sum(np.logical_and(self.predict(X) == 0, y == 1))
            return 2 * tp / (2 * tp + fp + fn)
        else:
            raise ValueError('Metric not supported. Supported metrics: accuracy, precision, recall, f1')
    
    def __repr__(self):
        return f'Perceptron(learning_rate={self.learning_rate}, n_iter={self.n_iter})'
    
    def __str__(self):
        """
        Returns a string representation of the Perceptron object.

        Returns:
            str: A string representation of the Perceptron object.
        """
        return f'Perceptron(learning_rate={self.learning_rate}, n_iter={self.n_iter})'