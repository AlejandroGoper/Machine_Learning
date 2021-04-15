#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:56:03 2021

@author: Alejandro Gomez

Programa para implementar un arbol binario y sus recorridos principales

"""

class Nodo:
    """
        Atributos 
    """
    def __init__(self, valor):
        
        #Atributos para los nodos hijos
        self.izquierda = None 
        self.derecha = None
        self.valor = valor

"""
Subrutina insertar: 
    Busca recursivamente un lugar donde insertar un nodo determinado 
Parametros:
    nodo_padre
    nodo_a_insertar
"""
def insertar(nodo_padre,nodo_a_insertar):
    #Si el nodo padre es la raiz.
    if nodo_padre == None:
        #Insertamos el nodo
        nodo_padre = nodo_a_insertar
    #Sino
    else:
        if nodo_a_insertar.valor > nodo_padre.valor:
            #Verificamos si el nodo de la derecha esta vacio.
            if nodo_padre.derecha == None:
                #Insertamos el nodo
                nodo_padre.derecha = nodo_a_insertar
            else:
                #Si el nodo de la derecha no esta vacio, ahora el nodo
                #padre sera este nodo y repetimos la busqueda
                insertar(nodo_padre.derecha,nodo_a_insertar)
            #Si el valor del nodo a insertar es menor que el de su padre
            # entonces nos dirigimos al lado izquierdo del arbol
        else:
        #Verificamos si el nodo hijo de la izquierda esta vacio
            if nodo_padre.izquierda == None:
            #Insertamos
                nodo_padre.izquierda = nodo_a_insertar
            else:
                #Si no esta vacio, el nodo padre, sera este nodo hijo de
                # la izquierda y repetimos la busqueda
                insertar(nodo_padre.izquierda,nodo_a_insertar)
    
"""
 ====================== Subrutinas de recorridos: =========================
"""    


"""
    En orden:
        Imprime por pantalla los numeros ordenados de manera ascendente
        
    Parametros:
        nodo_padre: que generalmente sera la raiz del arbol
        lista: una lista donde se guardaran los valores del recorrido
"""
def en_orden(nodo_padre,lista):
    #Si el nodo padre es None no se hace nada, pero, sino:
    if nodo_padre != None:
        #Recorremos primero el nodo hijo de la izquierda
        en_orden(nodo_padre.izquierda,lista)
        #Imprimimos el valor del nodo en pantalla
        print(nodo_padre.valor)
        #Agrego el valor del nodo a la lista
        lista.append(nodo_padre.valor)
        #Visitamos el nodo hijo de la derecha
        en_orden(nodo_padre.derecha,lista)

"""
    Pre Orden:
        Imprime por pantalla los valores del nodo padre y de sus hijos 
        izquierdo y derecho
        
    Parametros:
        nodo_padre: que generalmente sera la raiz del arbol
        lista: una lista donde se guardaran los valores del recorrido
"""
def pre_orden(nodo_padre,lista):
    #Si el nodo no esta vacio
    if(nodo_padre != None):
        #Imprimo el valor del nodo
        print(nodo_padre.valor)
        #Agrego ese valor a la lista
        lista.append(nodo_padre.valor)
        #Visitamos a su hijo de la izquierda
        pre_orden(nodo_padre.izquierda,lista)
        #Visitamos a su hijo de la dercha
        pre_orden(nodo_padre.derecha,lista)

"""
    Post Orden:
        Imprime por pantalla los valores del nodo hijo izquierdo y derecho
        y por ultimo el de su padre
        
    Parametros:
        nodo_padre: que generalmente sera la raiz del arbol
        lista: una lista donde se guardaran los valores del recorrido
"""
        
def post_orden(nodo_padre,lista):
    #Si el nodo no esta vacio
    if(nodo_padre != None):
        #Visimatos al nodo hijo izquierdo
        post_orden(nodo_padre.izquierda,lista)
        #Visitamos al nodo hijo derecho
        post_orden(nodo_padre.derecha,lista)
        #Agregamos el valor a la lista
        lista.append(nodo_padre.valor)
        #Imprimo el valor del nodo
        print(nodo_padre.valor)

"""
    guardar_listas:    
        Función para guardar las listas de valores de cada recorrido a un 
        archivo con extension .csv utilizando un metodo de la libreria pandas
    Parametros:
        lista1: lista del recorrido en orden
        lista2: lista del recorrido pre orden
        lista3: lista del recorrido post orden
"""
def guardar_listas(lista1,lista2,lista3):
    #Importo un metodo de la libreria pandas que me permite crear un dataframe
    from pandas import DataFrame
    #Creo el data frame pasando como argumento una matriz M (lista de listas)
    # con una dimension de 3x32
    #con el parametro index especifico el nombre de las filas
    #con el parametro dtype especifico que los valores de M seran todos int
    df = DataFrame([lista1,lista2,lista3],index=["En orden:","Pre Orden:", "Post Orden:"],dtype=int)
    #Guardamos el data frame con el nombre de recorridos.csv
    # especificamos ademas que no guarde la numeracion por default de las columnas
    df.to_csv("recorridos.csv",header=False)
    
"""
    Programa Principal
"""
#Importamos metodos de la libreria numpy y random utiles
from numpy import loadtxt
from random import choice

#Cargamos los datos en la variable data
data = loadtxt("aleatorios.csv",delimiter=",")
#Redimensionamos la matriz de 8x4 a 32x1 
Matriz = data.reshape(32,1)
#Convertimos la matriz a una lista unidimensional
aleatorios = list(Matriz[:,0])
#Insertamos como nodo raiz, una posicion aleatoria del arbol
valor = choice(aleatorios)
raiz = Nodo(valor)
#Ya que tenemos definido el nodo raiz, insertamos los demás nodos.
# Pero eliminando el valor que ya insertamos del arreglo
aleatorios.pop(aleatorios.index(valor))
for numero in aleatorios:
    insertar(raiz,Nodo(numero))
#Imprimimos en pantalla el arreglo con distintos recorridos.
print(" ===== In-order ====\n")
lista1 = []
en_orden(raiz,lista1)
print(" ===== Pre-order ====\n")
lista2 = []
pre_orden(raiz,lista2)
print(" ===== Post-order ====\n")
lista3 = []
post_orden(raiz,lista3)
#Guardamos los valores en el archivo llamado recorridos.csv
guardar_listas(lista1,lista2,lista3)    