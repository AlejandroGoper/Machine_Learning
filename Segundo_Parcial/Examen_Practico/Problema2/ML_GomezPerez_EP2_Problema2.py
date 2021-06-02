#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:42:23 2021

@author: Israel Alejandro Gómez Pérez

Problema2: El dataset_4classes2D adjunto contiene 800 puntos en R2 distribuidos
en 4 clases distintas. 
A) Utilice validación cruzada de 10 pliegues para evaluar los clasificadores 5-
NN, SVM con kernel lineal (hiperparámetros de su elección) y Perceptron 
Multicapa. 
B) Genere una imagen con las fronteras de decisión que ob tuvo con cada uno de
sus clasificadores. 
C) Proporcione una tabla con los accuracies promedio obtenidos por cada
uno de los 3 clasificadores.

 
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
from sklearn.neural_network import MLPClassifier

""" Cargando archivos de datos """

ds_1 = read_csv("Archivos/dataset_4classes2D.csv")

ds1 = array(ds_1)

"""  Dando formato a los datasets """

x_1 = ds1[:,:-1] # Eliminando la columna de etiquetas
y_1 = ds1[:,-1] # Tomando solo la columna de las etiquetas 
""" Generando listas contenedoras """


datasets = [x_1]
etiquetas = [y_1]
clasificadores = [KNN(n_neighbors=5),
                  SVC(kernel="linear", C = 2),
                  #PMC(n_inputs=3, n_outputs=3,epochs=100,learning_rate=0.01), # Implementacion Propia
                  #Perceptron(tol=0.001,random_state=6), # Implementacions SKLearn
                  #PMLMC(dim_input=2, neurons_H=5, neurons_O=4,epochs=100,learning_rate=0.5)] # Implementaion propia
                  MLPClassifier(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(5,2),random_state=6)] # Implementacion sklearn
nombres = ["3-NN ", "SVC LINEAR", "Perceptron Mult Sklearn"]


""" Realizando las fronteras de decision """

fd = FronterasDeDesicion(datasets, etiquetas, clasificadores, nombres, clases=4)
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

df = DataFrame(accuracies,index=["4 clases"],columns=["C1","C2","C3"])
print(df)

print("=======================================================================")
print("C1: 5 - NN ")
print("C2: SVC LINEAR")
print("C3: Perceptron Multicapa SKlearn")

