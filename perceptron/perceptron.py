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
    """

    def __init__(self, learning_rate=0.01, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
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
        weighted_sum = np.dot(X, self.w_) + self.b_
        return self.activation(weighted_sum)
    
    def brute_predict(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)