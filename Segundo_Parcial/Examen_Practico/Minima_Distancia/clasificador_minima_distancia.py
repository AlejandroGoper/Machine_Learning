"""
    Autor: I. Alejandro Gómez

    Este código es para realizar la clasificacion por minima distacia.
    Basicamente recibe un dataset de entrenamiento, otro de prueba y las etiquetas, hace el promedio
    de todos los puntos de cada clase en el dataset entrenamiento para encontrar el centroide de cada
    grupo y luego para cada punto del dataset de prueba verifica de que centro está más cerca y a ese
    lo asigna como predicción
"""

from numpy import array,unique,where
from pandas import DataFrame

class MinimaDistancia():
    # Método constructor
    def __init__(self):
        self.centros_de_clase = None

    """
        Método Fit: Realiza el entrenamiento del clasificador

        Parámetros: 
            - training_ds: Datos de prueba
            - training_y_labels: Etiquetas de los datos de prueba
    """
    def fit(self,trainning_ds,trainning_y_labels):
        # Utilizando pandas para el filtrado de las clases
        df = DataFrame(trainning_ds,columns=["X","Y"])
        df["Label"] = trainning_y_labels
        # Obtengo en un arreglo las unicas clases diferentes
        clases = unique(trainning_y_labels)
        # Para cada clase realizo el promedio de todos sus valores en el dataset
        centroides = []
        for clase in clases:
            #Filtro el dataset por su valor de etiquetas
            df_filtrado = df[df["Label"] == clase]
            Sum_X = df_filtrado["X"].sum() #Sumo toda la columna de las componentes X
            Sum_Y = df_filtrado["Y"].sum() #Sumo toda la columna de las componentes Y
            n = len(df_filtrado["X"])
            centroides.append([Sum_X/n,Sum_Y/n]) #Guardo el punto promedio de cada clase en una lista
        self.centros_de_clase = array(centroides) # Convierto la lista a un arreglo numpy
    
    """
        Método Predict: Realiza las predicciones con el clasificador entrenado

        Parámetros:
            - test_ds: Data set de prueba
        Regresa:
            - predicted_labels: Array tipo numpy con las etiquetas predichas
    """
    def predict(self,test_ds):
        # Realizo la clasificación por minima distancia
        # Calculo la distancia de cada punto en test_ds a cada uno de los centroides
        predicted_labels = []
        centroides = self.centros_de_clase
        for punto in test_ds:
            diff = punto-centroides # Esto me genera una matriz con las distancia del punto pivote a cada centroide
            d2 = diff[:,0]*diff[:,0] + diff[:,1]*diff[:,1] # Calculo la distancia al cuadrado
            indice_valor_minimo = where(d2 == d2.min())
            indice_valor_minimo = int(indice_valor_minimo[0])
            predicted_labels.append(indice_valor_minimo)
        return array(predicted_labels)