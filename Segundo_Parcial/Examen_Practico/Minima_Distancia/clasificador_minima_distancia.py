#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:24:23 2021

@author: alejandro_goper

Clasificador por minima distancia valido para datasets con m�s de dos clases

Localiza los centroides de cada clase en la parte del entrenamiento, luego, 
para cada nuevo punto, calcula la distancia a cada centroide y asigna al nuevo
punto la etiqueta de la clase m�s cercana.

"""

""" Cargando modulos necesarios """

from numpy import array,unique,where
from pandas import DataFrame

class MDC():
    # Metodo constructor
    def __init__(self):
        # Atributo de clase
        self.centros_de_clase = None

    """
        Metodo Fit: Realiza el entrenamiento del clasificador

        Parametros: 
            - training_ds: Datos de entrenamiento
            - training_y_labels: Etiquetas de los datos de entrenamiento
    """
    def fit(self,trainning_ds,trainning_y_labels):
        # Utilizando pandas para el filtrado de las clases
        # Creamos un dataset con los valores de training_ds
        df = DataFrame(trainning_ds,columns=["X","Y"])
        # Agregamos una columna con las etiquetas correspondientes
        df["Label"] = trainning_y_labels
        
        # Obtengo en un arreglo las unicas clases diferentes ordenadas
        clases = unique(trainning_y_labels)
        # Para cada clase realizo el promedio de todos sus valores en el dataset
        centroides = []
        for clase in clases:
            #Filtro el dataset por su valor de etiquetas unicas de clase
            df_filtrado = df[df["Label"] == clase]
            Sum_X = df_filtrado["X"].sum() #Sumo toda la columna de las componentes X
            Sum_Y = df_filtrado["Y"].sum() #Sumo toda la columna de las componentes Y
            n = len(df_filtrado["X"]) # Como la dimension de X es igual a la de Y 
            centroides.append([Sum_X/n,Sum_Y/n]) #Guardo el punto promedio de cada clase en una lista
        
        #Agrego las coordenadas de los centroides como atributo de clase
        self.centros_de_clase = array(centroides) # Convierto la lista a un arreglo numpy
    
    """
        Metodo Predict: Realiza las predicciones con el clasificador entrenado

        Parametros:
            - test_ds: Data set de prueba
        Regresa:
            - predicted_labels: Array tipo numpy con las etiquetas predichas
    """
    def predict(self,test_ds):
        # Realizo la clasificacion por minima distancia
        # Calculo la distancia de cada punto en test_ds a cada uno de los centroides
        predicted_labels = []
        centroides = self.centros_de_clase
        for punto in test_ds:
            diff = punto-centroides # Esto me genera una matriz con las distancia del punto pivote a cada centroide
            d2 = diff[:,0]*diff[:,0] + diff[:,1]*diff[:,1] # Calculo la distancia al cuadrado
            # En el arreglo d2 encuentro el indice de su valor mas peque�o
            indice_valor_minimo = where(d2 == d2.min())
            # la linea anterior regresa una matriz de tipo float de dim [1x1]
            # lo convertimos simplemente a int
            indice_valor_minimo = int(indice_valor_minimo[0])
            # el indice del valor minimo corresponde con la clase predicha
            predicted_labels.append(indice_valor_minimo)
        return array(predicted_labels)