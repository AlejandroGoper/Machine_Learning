#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 01:12:20 2021

@author: alejandro_goper

Script para realizar validación cruzada de 10 pliegues 
"""

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from numpy import trace,array

class ValidacionCruzada():
    # Metodo constructor
    def __init__(self, datasets,etiquetas,clasificadores,nombres):
        self.datasets = datasets
        self.labels = etiquetas
        self.clasificadores = clasificadores
        self.nombres = nombres


    """
    Argumentos:
        pliegues: numero de pliegues para realizar la validación cruzada
        
    Regresa:
        accuracies_final_ds: arreglo del promedio de los k pliegues por cada clasificador por cada 
                                dataset.
    """
   
    def calcular(self,pliegues):
        
        datasets = self.datasets
        labels = self.labels
        clasificadores = self.clasificadores
        
        # Instanciamos la clase KFold
        kf = KFold(n_splits = pliegues)
        # arreglo que controla los accuracies a retornar
        accuracies_final_ds = []
        
        # Iteramos sobre los datasets
        for dataset,label,id_ds in zip(datasets,labels,range(len(datasets))):
            # Arreglo que controla los accuracies por pliego de todos los clasificadores
            accuracies_por_pliego = []
            # Realizamos los pliegues y comienza la validacion cruzada
            # De esta manera obtenemos los indices del conjunto de entrenamiento y prueba
            for train_index, test_index in kf.split(dataset):
                # Recuperamos el conjunto de entranamiento a partir de los indices
                train_ds = dataset[train_index]
                train_lbl = label[train_index]
                # Recuperamos el conjunto de prueba 
                test_ds = dataset[test_index]
                test_lbl = label[test_index]
                # Entrenamos todos los clasificadores con este conjunto
                # y luego predecimos 
                accuracies_clasificadores = []
                for clasificador in clasificadores:
                    clasificador.fit(train_ds,train_lbl)
                    predicted_lbl = clasificador.predict(test_ds)
                    # Realizo la matriz de confusión
                    cm = confusion_matrix(test_lbl,predicted_lbl)
                    """
                    # Solución Ad-hoc para el clasificador perceptron multicapa
                    if((clasificador == clasificadores[-1]) and id_ds == 3):
                        for i in range(len(predicted_lbl)):
                            predicted_lbl[i] = int(predicted_lbl[i] == 0)
                        cm = confusion_matrix(test_lbl,predicted_lbl,labels=(1,0))
                    else:
                        cm = confusion_matrix(test_lbl,predicted_lbl)
                    """
                    # Calculo el accuracy
                    acc = trace(cm)/cm.sum()
                    accuracies_clasificadores.append(acc)
                accuracies_por_pliego.append(accuracies_clasificadores)
            # Debemos realizar el promedio de los accuracies por pliego 
            accuracies_por_pliego = array(accuracies_por_pliego)
            # Arreglo de los promedios de los accuracies de cada clasificador por data set
            accuracies = []
            accuracies.append(accuracies_por_pliego[:,0].sum()/pliegues)
            accuracies.append(accuracies_por_pliego[:,1].sum()/pliegues)
            accuracies.append(accuracies_por_pliego[:,2].sum()/pliegues)
            accuracies.append(accuracies_por_pliego[:,3].sum()/pliegues)
            accuracies.append(accuracies_por_pliego[:,4].sum()/pliegues)
            # Añadir al arreglo final
            accuracies_final_ds.append(accuracies)
        return array(accuracies_final_ds)