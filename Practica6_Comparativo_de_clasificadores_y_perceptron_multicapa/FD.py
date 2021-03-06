"""
    Codigo tomado de: https://www.aprendemachinelearning.com/clasificar-con-k-nearest-neighbor-ejemplo-en-python/
    
    Se han modificado algunas lineas de codigo para adaptarlo al programa principal
    
"""

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from numpy import meshgrid,arange,c_
import matplotlib.patches as mpatches

class FronterasDeDesicion():
    # Metodo constructor 
    def __init__(self, datasets,etiquetas,clasificadores,nombres):
        self.datasets = datasets
        self.labels = etiquetas
        self.clasificadores = clasificadores
        self.nombres = nombres

    """
        Realiza una particion muy fina de elementos para poder graficar la frontera de desicion
        en función del clasificador.
        
        Parametros: 
            X: Dataset de prueba
            clf: objeto perteneciente al clasificador
        Regresa:
            Z: funcion de desición
            xx, yy: arreglo unidimensional de la coordenada x,y de la particion fina
    """    

    def hacer_grid(self,X,clf):
        
        h = 0.05
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = meshgrid(arange(x_min, x_max, h), arange(y_min, y_max, h))
        
        test = c_[xx.ravel(),yy.ravel()]
        
        Z = clf.predict(test)
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        
        return Z,xx,yy
    
    """
        Realiza el proceso de graficar y de entrenar los clasificadores 
        
    """
    
    def mostrar(self):
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#c2f0c2']) 
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
        # Para identificar cada region con un color         
        patch0 = mpatches.Patch(color='#FF0000', label='Clase 1')
        patch1 = mpatches.Patch(color='#00FF00', label='Clase 2')
        # atributos de clase 
        datasets = self.datasets
        labels = self.labels
        clasificadores = self.clasificadores
        nombres = self.nombres
        # creo una figura lo suficientemente grande 
        fig = plt.figure(figsize=(30,30))
        # pongo esto para que no se empalmen
        fig.tight_layout()
        
        # Iteramos sobre los datasets,labels y un identificador de las graficas de datos puros
        for dataset,label,id_ds in zip(datasets,labels,[6,12,18,24]):
            # id_ds es para ubicar las graficas de los datasets al final de cada renglon
            # tenemos un grid de 3 filas y 5 columnas
            ax = plt.subplot(4,6,id_ds)
            # grafico al final de cada renglon el dataset original
            ax.scatter(dataset[:,0],dataset[:,1],c=label,cmap='plasma',s=2)
            ax.set_title("Dataset original")
            # Ahora iteramos por clasificadores, nombre y un identificador de clasificador
            for clasificador,nombre,id_clf in zip(clasificadores,nombres,range(1,6)):
                # Entrenamos el clasificador
                clasificador.fit(dataset,label)
                # Realizamos la particion fina para graficar la frontera de desición
                Z,xx,yy = self.hacer_grid(dataset,clasificador)
                # Graficamos en el lugar deseado 
                ax = plt.subplot(4,6,(id_ds-6)+id_clf)
                # Graficamos la partción fina (frontera de desición) 
                ax.pcolormesh(xx,yy,Z,cmap=cmap_light)
                # Graficamos alli mismo el dataset original 
                ax.scatter(dataset[:,0],dataset[:,1],c=label,cmap = cmap_bold,s=2)
                # Definimos dominio y rango de la grafica
                ax.set_xlim(xx.min(),xx.max())
                ax.set_ylim(yy.min(),yy.max())
                # Agregamos la leyenda para identificar las clases 
                ax.legend(handles=[patch0,patch1])
                # Agregamos el titulo correspondiente a cada grafica
                ax.set_title(f"clasificador: {nombre} ")
                