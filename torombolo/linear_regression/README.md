# Regresión Lineal

<!-- ![Regresión Lineal](regresion_lineal_animation.gif) -->

La regresión lineal es una técnica de aprendizaje automático utilizada para modelar y predecir relaciones lineales entre una variable dependiente y una o más variables independientes. Este repositorio proporciona ejemplos y código para entender y aplicar la regresión lineal en varios escenarios.

# Instrucciones de Uso

## Regresión Lineal

Para usar el modelo de regresión lineal, necesitas importar la clase `LinearRegression`. Puedes hacerlo de las siguientes maneras:

option 1: importar la clase `LinearRegression` desde el paquete `torombolo`

```python
from torombolo import LinearRegression
```

option 2: importar la clase `LinearRegression` desde el módulo `linear_regression`

```python
from torombolo.linear_regression import LinearRegression
```

Una vez importada la clase `LinearRegression`, puedes crear una instancia de la clase y usarla para entrenar un modelo de regresión lineal. Para ello, debes pasarle los datos de entrenamiento a los parámetros `X` y `y` de la función `fit`. Por ejemplo:

```python
from torombolo import LinearRegression

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X, y)
```

Una vez entrenado el modelo, puedes usarlo para hacer predicciones. Para ello, debes pasarle los datos de prueba a la función `predict`. Por ejemplo:

```python
from torombolo import LinearRegression

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146]

model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X, y)

X_test = [11, 12]
predictions = model.predict(X_test)
```

para ver el intercepto y los coeficientes del modelo, puedes usar los atributos `intercept_` y `slope_` de la clase `LinearRegression`. Por ejemplo:

```python
from torombolo import LinearRegression

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146]

model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X, y)

print(model.intercept_)
print(model.slope_)
```