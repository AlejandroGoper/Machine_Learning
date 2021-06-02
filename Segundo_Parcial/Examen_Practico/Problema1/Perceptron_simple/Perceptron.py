#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:51:55 2021

@author: alejandro_goper

Modificación de la clase Perceptron proporcionada en clase
para ajustarse al codigo main.py 
"""

from numpy import array

class Perceptron():
    """
        Implementación simple del algoritmo perceptron con función
        de activación escalon.
        
        Basada en el script proporcionado en clase "1 Perceptron.ipynb"
    """
    def __init__(self,w0=1,w1=0.1,w2=0.1):
        # weights 
        self.w0 = w0 # bias
        self.w1 = w1
        self.w2 = w2

    def step_function(self,z):
       if (z >= 0):
           return 1
       else:
           return 0
    
    def weighted_sum_inputs(self, x1, x2):
        return sum([1*self.w0, x1*self.w1,x2*self.w2])
    
    def predict(self,test_ds):
        predicted_labels = []
        for point in test_ds:
            x1 , x2 = point[0], point[1] 
            z = self.weighted_sum_inputs(x1, x2)
            label = self.step_function(z)
            predicted_labels.append(label)
        return array(predicted_labels)
    
    """ Step es la taza de aprendizaje"""
    def fit(self, X, y, epochs = 1, step = 0.1):
        errors = []
        for epoch in range(epochs):
            error = 0.0
            for point, target in zip(X,y):
                x1 , x2 = point[0], point[1] 
                # La actualización es proporcional al tamaño de paso (step) y al error 
                update = step*(target - self.predict([point]))
                self.w1 += update*x1
                self.w2 += update*x2
                self.w0 += update
                error += int(update != 0.0)
            errors.append(error)