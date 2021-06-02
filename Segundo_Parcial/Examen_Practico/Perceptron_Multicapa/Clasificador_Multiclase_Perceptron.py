#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:43:48 2021

@author: alejandro_goper

Se realizará un implementacion para realizar clasificación de varias
clases con la regla de aprendizaje del perceptron, para ello, se necesitara
tantas neuronas de salida como etiquetas queramos clasificar

Utilizaremos el metodo de descenso del gradiente con la funcion de perdida
CROSS ENTROPY (vease: https://juansensio.com/blog/017_clasificacion_multiclase)
y la funcion de activacion SOFTMAX el programa tomara el valor mayor entre las 
salidas de esta funcion como la prediccion.


Codigo basado en el link de arriba.
"""

import numpy as np

# Nombre de la clase derivado de Perceptron Multiclass Classifier
class PMC:
    def __init__(self, n_inputs , n_outputs):
        # Ahora habran n_inputs pesos asociados a cada neurona de salida
        # lo cual es una matriz de n_inputs x n_outputs de pesos 
        self.w = np.random.rand(n_inputs,n_outputs)
     
        # Atributos para normalizacion de puntos nuevos a predecir
        self.x_mean = None
        self.x_std = None
    
    """
        Metodo: Funcion de activacion
        
        Softmax
    """
    def activacion(self, z):
        return np.exp(z)/np.exp(z).sum()   
    
    """
        Metodo: d_loss 
            
        Derivada de la funcion de perdida
        
    """
    def d_loss(self, output, target):
        # Creamos una matriz de ceros de las mismas dimensiones que output
        resultado = np.zeros_like(output)
        # Ponemos 1 en posiciones especificas
        resultado[np.arange(len(output)),target] = 1
        return (-resultado + self.activacion(output))/output.shape[0]
    
    """
        Metodo: Neuron output
        Controla la salida de la neurona, es decir, realiza la suma ponderada
        y la pasa a la funcion de activacion.
    """
    def neuron_output(self, x):
        # esto me devuelve un array con la suma ponderada para cada neurona
        suma_ponderada = np.dot(x,self.w)
        return suma_ponderada
    
    
    def predict(self, x_test):
        # Primero debemos transformar los datos a probabilidades
        x_prob = (x_test - self.x_mean)/self.x_std
        # Agregamos una columa de unos para el BIAS
        x_prob = np.c_[np.ones(len(x_prob)),x_prob]
        # hacemos la predicción
        y_out = self.neuron_output(x_prob) 
        # pasamos la suma ponderada por la funcion de activacion
        y_output = self.activacion(y_out)
        # las etiquetas predichas seran el indice de los elementos maximos
        # de la clase
        y_pred = np.argmax(y_output,axis=1)
        return y_pred
        
    """
        Metodo: Fit
        
        Entrenamiento de la red
    """
    def fit(self,x_train,y_train,epochs = 15, learning_rate = 0.1):
        
        self.x_mean = x_train.mean(axis=0)
        self.x_std = x_train.std(axis=0)
        
        # Agregamos una columna de unos para el BIAS
        x_train = np.c_[np.ones(len(x_train)),x_train]
        # Realizamos el aprendizaje por cada epoca
        for epoch in range(epochs):
            y_pred = self.neuron_output(x_train)
            # Derivada de la funcion de perdida
            dl_dh =  self.d_loss(y_pred, y_train)
            dh_dw = x_train # porque h = w*x
            # Derivada de la funcion de perdida con respecto a los pesos
            # aplicamos regla de la cadena dl_dw = dl_dh*dh_dw
            
            # Aplicamos el producto punto porque la multiplicacion se hace 
            # posicion por posicion y luego sumamos todas las contribuciones 
            dl_dw = np.dot(dh_dw.T,dl_dh)
            
            # Actualizamos los pesos 
            self.w = self.w - learning_rate*dl_dw