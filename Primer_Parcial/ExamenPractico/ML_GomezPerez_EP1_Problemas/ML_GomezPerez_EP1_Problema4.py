# -*- coding: utf-8 -*-
"""
Problema 4: Se tomo como referencia este código;
Se ha modificado la función find_winner de modo que imprima cada vez que se 
llama, el atributo con mayor ganancia de información así como el valor de la
ganancia de información.

Created on Mon Feb 22 13:31:06 2021

@author: TUF-PC8

https://medium.com/@lope.ai/decision-trees-from-scratch-using-id3-python-coding-it-up-6b79e3458de4
"""

import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log


#Definiendo el dataset
supplies = 'Low,High,Med,Low,Low,High,High,Med,Low,Low,Med,High'.split(',')
weather = 'Sunny,Sunny,Cloudy,Raining,Cloudy,Sunny,Raining,Cloudy,Raining,Raining,Sunny,Sunny'.split(',')
worked = 'Yes,Yes,Yes,Yes,No,No,No,Yes,Yes,No,No,Yes'.split(',')
shopped = 'Yes,No,No,No,Yes,No,No,No,No,Yes,Yes,No'.split(',')

#Creando Dataframe de pandas
dataset ={'supplies':supplies,'weather':weather,'worked':worked,'shopped':shopped}
df = pd.DataFrame(dataset,columns=['supplies','weather','worked','shopped'])

#1. calculando entropía del dataset completo
entropy_node = 0  #Initialize Entropy
values = df.shopped.unique()  #Unique objects - 'Yes', 'No'
for value in values:
    fraction = df.shopped.value_counts()[value]/len(df.shopped)  
    entropy_node += -fraction*np.log2(fraction)


#2. Función para calcular la entropía de cada atributo    
def ent(df,attribute):
    target_variables = df.shopped.unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Sweet')

    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df.shopped ==target_variable]) #numerator
            den = len(df[attribute][df[attribute]==variable])  #denominator
            fraction = num/(den+eps)  #pi
            entropy_each_feature += -fraction*log(fraction+eps) #This calculates entropy for one feature like 'Sweet'
        fraction2 = den/len(df)
        entropy_attribute += -fraction2*entropy_each_feature   #Sums up all the entropy ETaste

    return(abs(entropy_attribute))
    
#Guardar la entropía de cada atributo
a_entropy = {k:ent(df,k) for k in df.keys()[:-1]}

# 3. Calculando la ganancia de información de cada atributo
def ig(e_dataset,e_attr):
    return(e_dataset-e_attr)

#entropy_node = entropy of dataset
#a_entropy[k] = entropy of k(th) attr
ig = {k:ig(entropy_node,a_entropy[k]) for k in a_entropy}

#Hasta este punto se calcula sólo para el atributo principal

#Usando recursión para terminar el árbol
def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    
    #Lineas adicionales para imprimir el atributo y la ganancia de información
    winner = df.keys()[:-1][np.argmax(IG)]
    print(f"IG de {winner} = {IG[np.argmax(IG)]}")
    return winner
  
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df,tree=None): 
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)
    #Get distinct value of that attribute 
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 
    
    for value in attValue:
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable[Class],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree

my_tree = buildTree(df)
import pprint
pprint.pprint(my_tree)