import numpy as np
import matplotlib.pyplot as plt

def generar_datos():
    X = np.random.rand(100, 2) * 100
    return X

def encontrar_vecinos(X, punto, epsilon):
    vecinos = []
    for i in range(len(X)):
        if np.linalg.norm(X[i] - punto) < epsilon:
            vecinos.append(i)
    return vecinos

def expandir_cluster(X, clasificaciones, punto_id, vecinos, cluster_id, epsilon, min_vecinos):
    clasificaciones[punto_id] = cluster_id
    i = 0
    while i < len(vecinos):
        vecino_id = vecinos[i]
        if clasificaciones[vecino_id] == -1:
            clasificaciones[vecino_id] = cluster_id
        elif clasificaciones[vecino_id] == 0:
            clasificaciones[vecino_id] = cluster_id
            vecinos_punto = encontrar_vecinos(X, X[vecino_id], epsilon)
            if len(vecinos_punto) >= min_vecinos:
                vecinos = vecinos + vecinos_punto
        i += 1

def dbscan(X, epsilon, min_vecinos):
    clasificaciones = [0] * len(X)
    cluster_id = 1
    for i in range(len(X)):
        if clasificaciones[i] == 0:
            vecinos = encontrar_vecinos(X, X[i], epsilon)
            if len(vecinos) < min_vecinos:
                clasificaciones[i] = -1
            else:
                expandir_cluster(X, clasificaciones, i, vecinos, cluster_id, epsilon, min_vecinos)
                cluster_id += 1
    return clasificaciones

X = generar_datos()

epsilon = 10.0
min_vecinos = 5
clusters = dbscan(X, epsilon, min_vecinos)

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow', marker='o')
plt.title("Clustering basado en densidad simple")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
