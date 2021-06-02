#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:42:23 2021

@author: Israel Alejandro Gómez Pérez

Problema3: El dataset_3classes4D adjunto contiene 150 puntos en R4 distribuidos
en 3 clases distintas (50 puntos por clase). 
A) Utilice el método PCA para encontrar las primeras 2 componentes principales 
de los datos proporcionados en el dataset. 
B) Grafique un scatterplot con la información de las 2 componentes principales 
encontradas. 
C) Seleccione 3 clasificadores supervisados y sus hiperparámetros para resolver
el dataset mapeado al espacio de 2 componentes principales. 
D) Genere una imagen con las fronteras de decisión obtenidas por sus 3 
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

# modulos alternativos
#from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

""" Cargando archivos de datos """

ds_1 = read_csv("Archivos/dataset_3classes4D.csv")

ds1 = array(ds_1)

"""  Dando formato a los datasets """

x_1 = ds1[:,:-1] # Eliminando la columna de etiquetas
y_1 = ds1[:,-1] # Tomando solo la columna de las etiquetas 

""" Aplicando PCA para obtener solo dos componentes principales """

pca = PCA(n_components=2)

pca.fit(x_1)

print("============================================================")
print(" Componentes principales: ")
print(pca.components_)
print("============================================================")

""" Transformando datos """
x_t_1 = pca.transform(x_1)

""" Graficando en el espacio transformado """

plt.figure()
plt.title("Grafico del espacio transformado")
plt.scatter(x_t_1[:,0],x_t_1[:,1], c= y_1, cmap= 'Set1')
plt.xlabel("x_t")
plt.ylabel("y_t")
plt.show()
""" Generando listas contenedoras """


datasets = [x_t_1]
etiquetas = [y_1]
clasificadores = [KNN(n_neighbors=7),
                  SVC(kernel="poly", C = 2,degree = 3),
                  MLPClassifier(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(5,3),random_state=6)] # Implementacion sklearn
nombres = ["7-NN ", "SVC poly", "Perceptron Mult Sklearn"]


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

df = DataFrame(accuracies,index=["3 clases PCA"],columns=["C1","C2","C3"])
print("**********************************************************************")
print(df)
print("**********************************************************************")
print("C1 -> 7 - NN")
print("C2 -> SVC Poly ")
print("C3 -> Perceptron Mult SKlear")