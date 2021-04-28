#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:26:12 2021

@author: alejandro_goper

Comparador de clasificadores:
    -Minima Distancia
    -K Nearest Neighboors
    -SVC RBF
"""
from numpy import genfromtxt
from clasificador_minima_distancia import MinimaDistancia
from FD import FronterasDeDesicion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Importamos los archivos .csv y arreglamos en la forma tradicional
dataset1 = genfromtxt("Archivos/dataset_classifiers1.csv",delimiter=',',skip_header=1)
dataset2 = genfromtxt("Archivos/dataset_classifiers2.csv",delimiter=',',skip_header=1)
dataset3 = genfromtxt("Archivos/dataset_classifiers3.csv",delimiter=',')
#Eliminamos la primer y ultima columna del dataset para quitar los indices y las etiquetas
X_1 = dataset1[:,1:-1] # Usamos slicing
X_2 = dataset2[:,1:-1]
X_3 = dataset3[:,:-1] # Aqui solo quitamos la ultima columna dado que no tiene la primera
#Asignando sus respectivas etiquetas a cada dataset 
y_true_1 = dataset1[:,-1]
y_true_2 = dataset2[:,-1]
y_true_3 = dataset3[:,-1]

# Creamos una lista contenedora de los datasets y etiquetas para iterar sobre ellos
datasets = [X_1,X_2,X_3]
labels = [y_true_1,y_true_2,y_true_3]



# Creamos una figura con un tamaño específico
#fig = plt.figure(figsize = (10,15))
# Para no tener empalmes en las graficas
#fig.tight_layout()
# Iteramos sobre los datasets, etiquetas y sobre un identificador de grafico
#for dataset,label,i in zip(datasets,labels,range(1,4)):
    # especificamos que haremos un grid de 3 filas 1 columna
 #   ax = plt.subplot(3,1,i)
    # Graficamos en cada espacio del grid
  #  ax.scatter(dataset[:,0],dataset[:,1],c=label,cmap='plasma',s=2)
   # ax.set_title(f"Dataset {i}")
#plt.show()


# Entrenando clasificadores

clasificadores = [MinimaDistancia(),KNeighborsClassifier(n_neighbors=5),SVC(kernel="rbf",C = 10, gamma=0.1)]

nombres = ["Minima Distancia", "KNeighborsClassifier","SVC RBF"]

fd = FronterasDeDesicion(datasets, labels, clasificadores, nombres)
fd.mostrar()