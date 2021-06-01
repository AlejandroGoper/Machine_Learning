#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:26:12 2021

@author: alejandro_goper

Comparador de clasificadores:
    -Minima Distancia
    -K Nearest Neighboors
    -SVC RBF
    -Perceptron
    -Perceptron Multicapa    
    
"""
from numpy import genfromtxt,array
from clasificador_minima_distancia import MinimaDistancia
from Perceptron import Perceptron
from FD import FronterasDeDesicion
from VC import ValidacionCruzada
from PerceptronMulticapa import PerceptronMulticapa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.linear_model import Perceptron as Per
from pandas import DataFrame,concat
from numpy.random import seed,normal,shuffle


# Importamos los archivos .csv y arreglamos en la forma tradicional
dataset1 = genfromtxt("Archivos/dataset_classifiers1.csv",delimiter=',',skip_header=1)
dataset2 = genfromtxt("Archivos/dataset_classifiers2.csv",delimiter=',',skip_header=1)
dataset3 = genfromtxt("Archivos/dataset_classifiers3.csv",delimiter=',')
#Eliminamos la primer y ultima columna del dataset para quitar los indices y las etiquetas
x_1 = dataset1[:,1:-1] # Usamos slicing
x_2 = dataset2[:,1:-1]
x_3 = dataset3[:,:-1] # Aqui solo quitamos la ultima columna dado que no tiene la primera
#Asignando sus respectivas etiquetas a cada dataset 
y_true_1 = dataset1[:,-1]
y_true_2 = dataset2[:,-1]
y_true_3 = dataset3[:,-1]

# Agregamos un cuarto dataset no separable (basado en la compuerta XOR)
"""
    Codigo basado en el script visto en clase "2 FFNN.ipynb"
"""

#Iniciando numeros aleatorios
seed(11)

# Creando el dataser XOR

# Media y desviacion estandar de las x pertenecientes a la primera clase
mu_x, sigma_x = 0, 0.1


# Creando la distribucion

d1 = DataFrame({'x1':normal(mu_x,sigma_x,200)+1,'x2':normal(mu_x,sigma_x,200)+1,'type':0})
d2 = DataFrame({'x1':normal(mu_x,sigma_x,200)+1,'x2':normal(mu_x,sigma_x,200)-1,'type':1})
d3 = DataFrame({'x1':normal(mu_x,sigma_x,200)-1,'x2':normal(mu_x,sigma_x,200)-1,'type':0})
d4 = DataFrame({'x1':normal(mu_x,sigma_x,200)-1,'x2':normal(mu_x,sigma_x,200)+1,'type':1})
dataset4 = array(concat([d1,d2,d3,d4],ignore_index=True))
shuffle(dataset4) # desordenando datos del dataset


x_4 = dataset4[:,:-1] # Eliminando la ultima columna
y_true_4 = dataset4[:,-1] # Tomamos solo la ultima columna

# Creamos listas contenedoras
datasets = [x_1,x_2,x_3,x_4]
labels = [y_true_1,y_true_2,y_true_3,y_true_4]

#datasets = [x_4]
#labels = [y_true_4]

clasificadores = [MinimaDistancia(),KNeighborsClassifier(n_neighbors=5),
                  SVC(kernel="rbf",C = 10, gamma=0.1),Perceptron(w0=1,w1=0.1, w2=0.1),
                  PerceptronMulticapa()]

nombres = ["Minima Distancia", "KNN","SVC RBF", "Perceptron", "Perceptron Multicapa"]

# Realizando grafico de fronteras de desición 

fd = FronterasDeDesicion(datasets, labels, clasificadores, nombres)
fd.mostrar()

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

df = DataFrame(accuracies,index=["linealmente separable", "anillos concentricos", "lunas","XOR"],columns=["C1","C2","C3","C4","C5"])
print(df)

print("=======================================================================")
print("C1: Minima Distancia")
print("C2: KNN")
print("C3: SVC RBF")
print("C4: Perceptron")
print("C5: Perceptron Multicapa")