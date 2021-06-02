#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 01:37:41 2021

@author: alejandro_goper

Implementacion del perceptron multicapa con algoritmo backpropagation para
clasificacion de mas de dos clases.

Se utilizaran la funcion de perdida Cross Entropy, y las funciones de 
activacion softmax para la ultima capa y relu para las capas ocultas.

Codigo basado en: https://juansensio.com/blog/024_mlp_clasificacion

"""
import numpy as np

# Nombre proveniente de las siglas Perceptron Multi Layer Multi Classifier
class PMLMC:
    def __init__(self, dim_input,neurons_H, neurons_O):
        # Inicializando pesos de la capa 1 de manera aleatoria con dist. normal
        self.w1 = np.random.normal(loc=0.0,scale=np.sqrt(2/(dim_input+neurons_H)),size=(dim_input,neurons_H))
        # Inicializando BIAS de la capa 1 en 0
        self.b1 = np.zeros(neurons_H)
        # Inicializando pesos de la capa 2 de manera aleatoria con dist. normal
        self.w2 = np.random.normal(loc=0.0,scale=np.sqrt(2/(neurons_H+neurons_O)),size=(neurons_H,neurons_O))
        # Inicializando BIAS de la capa 1 en 0
        self.b2 = np.zeros(neurons_O)
        
        # Atributos para calculos intermedios entre capas
        
        # controla la suma ponderada antes de la funcion de
        # activacion relu en la capa oculta
        self.h_pre = None 
        
        # controla la salida luego de la funcion de act. relu de la capa oculta
        self.h = None
        
        # Atributos para normalizacion de puntos nuevos a predecir
        self.x_mean = None
        self.x_std = None
    
    """
        Metodo: activacion
        
        Funcion de activacion identidad
    """
    def activacion(self, z):
        return z
    
    """
        Metodo: relu
        
        Funcion de activacion relu
    """
    def relu(self, z):
        return np.maximum(0,z)
    
    """
        Metodo: relu_prime
        
        Derivada de la funcion de activacion relu
    """
    def relu_prime(self, z):
        # devuelve 1 si z es positiva y 0 si es negativa
        return z > 0
    
    """
        Metodo: sofmax
        
        Funcion de activacion softmax
    """
    def softmax(self, z):
        return np.exp(z) / np.exp(z).sum(axis=-1,keepdims=True)
    
    """
        Metodo: d_loss 
            
        Derivada de la funcion de perdida
        
   """
    def d_loss(self, output, target):
        # Creamos una matriz de ceros de las mismas dimensiones que output
        resultado = np.zeros_like(output)
        # Ponemos 1 en posiciones especificas
        resultado[np.arange(len(output)),target] = 1
        return (-resultado + self.softmax(output))/output.shape[0]
    
    """
        Metodo: neuron_output
        
        Calcula la salida a la red, es decir, multiplica por los pesos de
        las capas.
    """
    
    def neuron_output(self, x):
         # salida de la capa 1
         # Multiplicamos los pesos de la primera capa por las entradas - x
         # y le sumamos el bias
         self.h_pre = np.dot(x, self.w1) + self.b1
         # Pasamos esta suma ponderada a la funcion de activacion de la capa 
         # oculta
         self.h = self.relu(self.h_pre)
         
         # Calculamos la salida total
         # Multiplicamos los pesos de la segunda capa por las entradas - h
         # y le sumamos el bias
         y_hat = np.dot(self.h, self.w2) + self.b2
         
         # mandamos a la funcion de activacion lineal de la ultima capa
         return self.activacion(y_hat)
     
    """
         Metodo: Fit
         
         Realiza entrenamiento de la red con el algoritmo backpropagation
    """   
     
    def fit(self, x_train,y_train,epochs=100,learning_rate=0.5):
        # Calculando la media y desviacion estandar de cada dimension
        self.x_mean = x_train.mean(axis=0)
        self.x_std = x_train.std(axis=0)
        
        x_prob = (x_train - self.x_mean)/self.x_std
        
        for epoch in range(epochs):
            # calculamos salida del perceptron
            y_pred = self.neuron_output(x_prob)
            # Backpropagation
            dldy = self.d_loss(y_pred, y_train) 
            dl_w2 = np.dot(self.h.T, dldy)
            dl_b2 = dldy.mean(axis=0)
            dldh = np.dot(dldy, self.w2.T)*self.relu_prime(self.h_pre)      
            dl_w1 = np.dot(x_prob.T, dldh)
            dl_b1 = dldh.mean(axis=0)
            # Actualizando pesos
            self.w1 = self.w1 - learning_rate * dl_w1
            self.b1 = self.b1 - learning_rate * dl_b1
            self.w2 = self.w2 - learning_rate * dl_w2
            self.b2 = self.b2 - learning_rate * dl_b2
     
    
    """
        Metodo: Predict
        
        predice etiquetas de clase posterior al entrenamiento del modelo
    """
    def predict(self,x_test):
        # Primero debemos transformar los datos a probabilidades
        x_prob = (x_test - self.x_mean)/self.x_std
        # hacemos la predicción
        y_out = self.neuron_output(x_prob) 
        # pasamos el resultado por la funcion de activacion softmax
        y_output = self.softmax(y_out)
        # las etiquetas predichas seran el indice de los elementos maximos
        # de la clase
        y_pred = np.argmax(y_output,axis=1)
        return y_pred