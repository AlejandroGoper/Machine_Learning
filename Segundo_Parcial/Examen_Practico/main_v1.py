#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:05:58 2021

@author: Israel Alejandro Gómez Pérez

Script principal para manejo de clasificadores: 2 Parcial Machine Learning

"""

""" Importando modulos """
from pandas import read_csv,DataFrame
from numpy import array
from Minima_Distancia.clasificador_minima_distancia import MDC
#from Perceptron_simple.Clasificador_Multiclase_Perceptron import PMC
#from Perceptron_Multicapa.Clasificador_Multiclase_Multicapa_Perceptron import PMLMC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from Fronteras_Desicion.FD import FronterasDeDesicion
from Validacion_Cruzada.VC import ValidacionCruzada
from sklearn.datasets import make_blobs
# modulos alternativos
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

""" Cargando archivos de datos """

ds_1 = read_csv("Archivos/dataset_classifiers1.csv")
ds_2 = read_csv("Archivos/dataset_classifiers2.csv")
ds_3 = read_csv("Archivos/dataset_classifiers3.csv")

ds1,ds2,ds3 = array(ds_1), array(ds_2), array(ds_3)

"""  Dando formato a los datasets """

""" Generando dataset 4 clases  para probar """

x_1,y_1 = make_blobs(n_samples=4000,centers=4,cluster_std=0.9,random_state=6)

""" Generando listas contenedoras """
datasets = [x_1]
etiquetas = [y_1]
clasificadores = [MDC(),KNN(n_neighbors=5),SVC(kernel="rbf", C = 10,gamma=0.1),
                  #PMC(n_inputs=3, n_outputs=4,epochs=100,learning_rate=0.001), # Implementacion Propia
                  Perceptron(tol=0.001,random_state=6), # Implementacions SKLearn
                  #PMLMC(dim_input=2, neurons_H=5, neurons_O=4,epochs=100,learning_rate=0.5)] # Implementaion propia
                  MLPClassifier(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(5,2),random_state=6)] # Implementacion sklearn
nombres = ["Minima Distancia", "KNN", "SVC RBF","Perceptron", "Perceptron Multicapa"]


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

df = DataFrame(accuracies,index=["4 clases"],columns=["C1","C2","C3","C4","C5"])
print(df)

print("=======================================================================")
print("C1: Minima Distancia")
print("C2: KNN")
print("C3: SVC RBF")
print("C4: Perceptron")
print("C5: Perceptron Multicapa")