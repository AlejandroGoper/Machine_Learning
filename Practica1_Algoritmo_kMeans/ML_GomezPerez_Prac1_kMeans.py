"""
    Autor: Israel Alejandro Gómez Pérez
    Programa: Algoritmo K-means
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import random

"""
    Esta función es para la asignación de cada punto con un centroide
    Parametros: 
    -centroides -- un arreglo numpy con los centroides de longitud k
    -conjunto_datos -- un arreglo bidimensional que contiene una lista de n puntos de dimensionalidad 2

    Retorno:
    - gamma -- matriz gamma
    - puntos_por_grupo -- un arreglo unidimensional de n puntos y cada elemento tiene el nummero de grupo al que pertenece  
"""

def asignar_grupo(centroides, conjunto_datos):
    gamma = np.zeros((len(conjunto_datos),len(centroides)))
    puntos_por_grupo = np.zeros((len(conjunto_datos),1))
    for i in range(len(conjunto_datos)):
        d = conjunto_datos[i] - centroides #Resta del punto pivote con el vector de centroides
        dd = d[:,0]*d[:,0] + d[:,1]*d[:,1] #distancia al cuadrado a cada uno de los centroides
        valor_minimo = np.amin(dd) #Valor minimo del vector distancias
        dd = list(dd)
        grupo = dd.index(valor_minimo) #El indice revela el grupo temporal más cercano al que pertenece
        puntos_por_grupo[i]=grupo
        gamma[i,grupo] = 1 #Ponemos 1 en la columna correspondiente a cada grupo más cercano
    return gamma,puntos_por_grupo


"""
    Esta función es para recalcular los centroides en el algoritmo K-means, realizando el promedio de los puntos obtenidos en cada grupo.
    Parámetros:
    - centroides -- un arreglo numpy con los centroides de longitud k 
    - conjunto_datos -- un arreglo bidimensional que contiene una lista de n puntos de dimensionalidad 2

    Retorno:
    - new_centroides -- un arreglo unidimensional de longitud k con los nuevos centroides.
"""

def actualizar_centroides(centroides, conjunto_datos, matriz_gamma):
    num_puntos_xgrupo = np.apply_along_axis(sum,0,matriz_gamma) #Tenemos en un vector el numero de puntos asociados a cada grupo temporal
    gamma_t = np.transpose(matriz_gamma) #Realizamos la transpuesta para luego multiplicar por el conjunto de datos y así sumar todos los elementos de un mismo grupo
    new_centroides = np.dot(gamma_t,conjunto_datos)
    for j in range(len(centroides)):
        new_centroides[j,:]*=(1/num_puntos_xgrupo[j]) #Dividimos la suma de los elementos del grupo j por el numero de puntos de ese grupo para sacar el promedio
    return new_centroides #Devolvemos los nuevos centroies.

"""
    Programa Principal
    -Genera dos gráficas, una con el dataset y otra con el resultado del algoritmo k-means
"""
# Creando el dataset
X, y_true = make_blobs(n_samples=400, centers= 4, cluster_std=.9,random_state=2)
plt.scatter(X[:,0],X[:,1],s=10)
plt.title("Data Set")
plt.savefig("Data_set.png") 

# Creamos matriz para los centroides y la matriz gamma para la identificación de los grupos
u = np.zeros((4,2))
gamma = np.zeros((400,4))
# Elegimos 4 centroides de manera aleatoria
for c in range(4):
    u[c] = random.choice(X)

# Ciclo principal
for t in range(10):
    gamma, puntos_por_grupo = asignar_grupo(centroides=u,conjunto_datos=X)        
    #Hasta aqui, tenemos ya construida la matriz gamma del sistema que asigna cada punto con su centroide mas cercano
    new_u = actualizar_centroides(centroides=u,conjunto_datos=X,matriz_gamma=gamma)
    u = new_u;

# Graficando mis resultados
plt.figure()
plt.scatter(X[:,0],X[:,1],c=puntos_por_grupo,s=10,cmap='plasma') 
centers = u
plt.scatter(centers[:,0],centers[:,1],c='green',s=50, alpha=1)
plt.title("Resultado algoritmo K-means")
plt.savefig("Results.png")
