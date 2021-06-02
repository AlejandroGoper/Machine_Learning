#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:59:53 2021

@author: Israel Alejandro Gómez Perez

Problema 1: El dataset_3classes2D.csv adjunto contiene 600 puntos en R2 
distribuidos en 3 clases distintas. 
A) Utilice validación cruzada de 10 pliegues para evaluar los clasificadores 3-
NN, SVM con kernel radial (hiperparámetros de su elección) y Perceptron. 
B) Genere una imagen con las fronteras de decisión que ob tuvo con cada uno de
 sus clasificadores.
C) Proporcione una tabla con los accuracies promedio obtenidos por cada uno de los 3
clasificadores 

"""


""" Importando modulos """
from pandas import read_csv,DataFrame
from numpy import array
#from Minima_Distancia.clasificador_minima_distancia import MDC
#from Perceptron_simple.Clasificador_Multiclase_Perceptron import PMC
#from Perceptron_Multicapa.Clasificador_Multiclase_Multicapa_Perceptron import PMLMC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from Fronteras_Desicion.FD import FronterasDeDesicion
from Validacion_Cruzada.VC import ValidacionCruzada
#from sklearn.datasets import make_blobs

# modulos alternativos
from sklearn.linear_model import Perceptron
#from sklearn.neural_network import MLPClassifier

""" Cargando archivos de datos """

ds_1 = read_csv("Archivos/dataset_3classes2D.csv")

ds1 = array(ds_1)

"""  Dando formato a los datasets """

x_1 = ds1[:,:-1] # Eliminando la columna de etiquetas
y_1 = ds1[:,-1] # Tomando solo la columna de las etiquetas 
""" Generando listas contenedoras """


datasets = [x_1]
etiquetas = [y_1]
clasificadores = [KNN(n_neighbors=3),
                  SVC(kernel="rbf", C = 10,gamma=0.1),
                  #PMC(n_inputs=3, n_outputs=3,epochs=100,learning_rate=0.01), # Implementacion Propia
                  Perceptron(tol=0.001,random_state=6) # Implementacions SKLearn
                  ]
                  #PMLMC(dim_input=2, neurons_H=5, neurons_O=4,epochs=100,learning_rate=0.5)] # Implementaion propia
                  #MLPClassifier(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(5,2),random_state=6)] # Implementacion sklearn
nombres = ["3-NN ", "SVC RBF", "Perceptron Sklearn"]


""" Realizando las fronteras de decision """

fd = FronterasDeDesicion(datasets, etiquetas, clasificadores, nombres, clases=3)
fd.mostrar()


""" Realizando validacion cruzada """

vd = ValidacionCruzada(datasets, etiquetas, clasificadores, nombres)
accuracies = vd.calcular(pliegues=10)

print ("\n\n")
print("-----------------------------------------------------------------------")
print("             Resultados: Validación cruzada de 10 pliegues")
print("                   Tabla de accuracies por clasificador   ")
print("                      por: I. Alejandro Gómez Pérez ")
print("-----------------------------------------------------------------------")
print("\n")

""" Mostrando la matriz de confusion """

df = DataFrame(accuracies,index=["3 clases"],columns=["C1","C2","C3"])
print(df)

print("=======================================================================")
print("C1: 3 - NN ")
print("C2: SVC RBF")
print("C3: Perceptron SKlearn")

