#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:14:19 2021

@author: alejandro_goper

Reimplementación del algoritmo perceptron para clasificación binaria
Utilizamos el algoritmo del desenso del gradiente y tomando como referencia
las ideas de codigo vistas en: https://www.youtube.com/watch?v=G9nQ6fbDkwY


"""

import numpy as np

# Por sus siglas de Perceptron Binary Classifier
class PBC():
    
    def __init__(self, size_w):
        # El parametro size_w es el tamaño de entradas del perceptron
        # Seran el numero de caracteristicas + el BIAS 
        # Inicializamos un vector de size_w numeros aleatorios 
        self.w = np.random.randn(size_w)
        # Designaremos el w[0] para el BIAS
        
        # Lista vacia para guardar todos los pesos y realizar visualizaciones
        self.ws = []
        
    """
        Metodo: Funcion de activacion
        
        Escalon unitario
    """    
        
    def activacion(self, z):
        if( z > 0):
            return 1
        else:
            return 0
        
    """
    Metodo predict 
    
    Devuelve la salida de la neurona, es decir, realiza la suma ponderada
    y la envia a la funcion de activacion, para datos no entrenados y sin la
    columna del BIAS
    
    """
    def predict(self, x_test):
        # Agregamos una columa de unos al dataset de entrada para agregar asi
        # el termino del BIAS
        x_test = np.c_[np.ones(len(x_test)),x_test]
        return self.neuron_output(x_test)
    """ 
        Metodo: neuron_output
        calcula la salida de la neurona para datos con la columna del BIAS 
        agregada
    """
    def neuron_output(self, x_test):
        suma_ponderada = np.dot(x_test,self.w)
        # Funcion de activacion escalon unitario
        pred = []
        for element in suma_ponderada:
            pred.append(self.activacion(element))
        return np.array(pred)
    
    
    def fit(self,x_train,y_train, epochs = 10, learning_rate = 0.1):
        # Agregamos una columa de unos al dataset de entrada para agregar asi
        # el termino del BIAS
        x_train = np.c_[np.ones(len(x_train)),x_train]
        # Entrenamos para cada epoca
        
        # Utilizaremos el algoritmo de descenso del gradiente usando como 
        # funcion de perdida el error cuadratico medio MSE
        # Vease https://www.youtube.com/watch?v=ytzyN1v7-6M
        for epoch in range(epochs):
            # Calculamos primero la salida de la neurona
            y_pred = self.neuron_output(x_train)
            # FUncion de perdida MSE l = 1/N sum( (y_pred - y_true)**2 )
            n = len(y_train)
            dl_dypred = (2/n)*(y_pred - y_train)
            dypred_dw = x_train # porque y_pred = w*x
            # Derivada de la funcion de perdida con respecto a los pesos
            # aplicamos regla de la cadena dl_dw = dl_dypred*dypred_dw
            
            # Aplicamos el producto punto porque la multiplicacion se hace 
            # posicion por posicion y luego sumamos todas las contribuciones 
            dl_dw = np.dot(dl_dypred,dypred_dw)
            # Actualizamos los pesos 
            self.w = self.w - learning_rate*dl_dw