import numpy as np
import matplotlib.pyplot as plt

def inicializar_centroides(X, k):
    
    indices = np.random.permutation(X.shape[0])
    centroides = X[indices[:k]]
    return centroides

def asignar_clusters(X, centroides):
    
    distancias = np.sqrt(((X - centroides[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distancias, axis=0)

def recalcular_centroides(X, labels, k):
    
    nuevos_centroides = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return nuevos_centroides

def k_means(X, k, numero_de_iteraciones=100):
    
    centroides = inicializar_centroides(X, k)
    historial_centroides = [centroides]
    for _ in range(numero_de_iteraciones):
        labels = asignar_clusters(X, centroides)
        centroides = recalcular_centroides(X, labels, k)
        historial_centroides.append(centroides)
        
    return centroides, labels, historial_centroides


X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])


k = 2
centroides_finales, labels, historial_centroides = k_means(X, k)


plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroides_finales[:, 0], centroides_finales[:, 1], color='black')
plt.title("K-means desde cero")
plt.show()
