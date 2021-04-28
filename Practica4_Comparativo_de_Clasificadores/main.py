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
from VC import ValidacionCruzada
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pandas import DataFrame


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

# Creamos listas contenedoras
datasets = [X_1,X_2,X_3]
labels = [y_true_1,y_true_2,y_true_3]
clasificadores = [MinimaDistancia(),KNeighborsClassifier(n_neighbors=5),SVC(kernel="rbf",C = 10, gamma=0.1)]
nombres = ["Minima Distancia", "KNeighborsClassifier","SVC RBF"]

# Realizando grafico de fronteras de desición 

#fd = FronterasDeDesicion(datasets, labels, clasificadores, nombres)
#fd.mostrar()

# Realizando validacion cruzada
vd = ValidacionCruzada(datasets, labels, clasificadores, nombres)
accuracies = vd.calcular(pliegues=10)

print ("\n\n")
print("-----------------------------------------------------------------------")
print("             Resultados: Validación cruzada de 10 pliegues")
print("                   Tabla de accuracies por clasificador   ")
print("                      por: I. Alejandro Gómez Pérez ")
print("-----------------------------------------------------------------------")
print("\n")

df = DataFrame(accuracies,index=["Dataset linealmente separable", "Dataset anillos concentricos", "Dataset lunas"],columns=nombres)
print(df)