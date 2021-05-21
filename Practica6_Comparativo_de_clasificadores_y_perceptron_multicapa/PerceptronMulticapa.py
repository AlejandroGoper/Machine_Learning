#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:12:02 2021

@author: alejandro_goper

Este Script es para implementar la clase perceptron multicapa basado en el
codigo propocionado en clase "2 FFNN.ipynb" adaptado para que pueda funcionar
con el resto de scripts de otros clasificadores

"""

from numpy.random import randn
from numpy import exp,append,ones,dot

class PerceptronMulticapa():
    """ Metodo constructor y definicion de atributos de clase"""
    def __init__(self, input_size = 2, hidden_size = 2, output_size = 1):
        # Sumando 1 a los que tendrán BIAS
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size
        
        # Atributos utiles para el algoritmo de backpropagation
        self.o_error = 0
        self.o_delta = 0
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.z2_error = 0
        self.z2_delta = 0
        
        # Construimos la matriz de pesos completa, desde las entradas hasta la capa oculta
        # con numeros aleotorios que siguen una distribucion normal 
        self.w1 = randn(self.input_size, self.hidden_size)
        # construimos el conjunto final de pesos desde la capa oculta hasta la salida
        self.w2 = randn(self.hidden_size, self.output_size)
    
    """ Función de activación """
    def sigmoide(self, s):
        return 1/(1+exp(-s))
    
    """ Derivada de la función de activación"""
    def sigmoide_prima(self, s):
        f = self.sigmoide(s)
        return f*(1-f)
    
    """ Propagación hacia adelante a través de la red"""
    def adelante(self, X):
        X = append(X,ones((len(X),1),axis=1)) # Agregamos 1 a las entradas para añadir el bias
        self.z1 = dot(X,self.w1) # Producto punto entre X (entradas) y el primer conjunto de 3x2 pesos
        self.z2 = self.sigmoide(self.z1) # Evaluamos el resultado en la función de activación
        self.z3 = dot(self.z2,self.w2) # Producto punto entre el resultado de las capas ocultas (z2) y el segundo conjunto de 3x1 pesos
        o = self.sigmoide(self.z3) # Funcion de activación final
        return o
    """ Algoritmo backpropagation para los errores """
    def atras(self,X,y,output,step):
        X = append(X,ones((len(X),1),axis=1)) # Agregamos 1 a las entradas para añadir el bias
        self.o_error = y - output # error en la salida
        self.o_delta = self.o_error*self.sigmoide_prima(output)*step # Aplicando la derivada de la sigmoide al error 
        self.z2_error = self.o_delta.dot(self.w2.T) # z2 error: que tanto contribuye al error de salida nuestra capa oculta de pesos
        self.z2_delta = self.z2_error * self.sigmoide_prima(self.z2)*step # Aplicando la derivada de la sigmoide al error z2
        self.w1 += X.T.dot(self.z2_delta) # Ajustando los primeros pesos
        self.w2 += self.z2.T.dot(self.o_delta) # Ajustando los segundos pesos
    
    """ Método Predict """
    def predict(self, X):
        return self.adelante(X)
    
    """ Método fit"""
    def fit(self, X, y,epochs = 10, step = 0.05):
        for epoch in range(epochs):
            X = append(X,ones((len(X),1),axis=1)) # Agregamos 1 a las entradas para añadir el bias
            output = self.adelante(X)
            self.atras(X, y, output, step)