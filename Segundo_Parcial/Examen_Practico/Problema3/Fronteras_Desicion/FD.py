"""
    Codigo tomado de: https://www.aprendemachinelearning.com/clasificar-con-k-nearest-neighbor-ejemplo-en-python/
    
    Se han modificado algunas lineas de codigo para adaptarlo al programa principal
    
"""

from matplotlib.colors import ListedColormap
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import meshgrid,arange,c_
import matplotlib.patches as mpatches

class FronterasDeDesicion():
    # Metodo constructor 
    def __init__(self, datasets,etiquetas,clasificadores,nombres,clases):
        self.datasets = datasets
        self.labels = etiquetas
        self.clasificadores = clasificadores
        self.nombres = nombres
        self.clases = clases
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
        cmap_set1 = cm.get_cmap("Set1").colors[:self.clases]
        patches = []
        for i in range(self.clases):
            patches.append(mpatches.Patch(color=cmap_set1[i], label="Clase {}".format(i)))

        #cmap_bold = ListedColormap(cm.get_cmap("Set1").colors[:self.clases])
        #colores_fondo = ['#FFAAAA', '#c2f0c2'] #  '#fbcf71', '#ffa264', '', '', ''
        #colores_puntos = ['#FF0000', '#00FF00'] # '#997619', '#a44c10', '', '', '' 
        #cmap_light = ListedColormap(colores_fondo) 
        #cmap_bold = ListedColormap(colores_puntos)
        # Para identificar cada region con un color         
        #patch0 = mpatches.Patch(color='#FF0000', label='Clase 1')
        #patch1 = mpatches.Patch(color='#00FF00', label='Clase 2')
        # atributos de clase 
        datasets = self.datasets
        labels = self.labels
        clasificadores = self.clasificadores
        nombres = self.nombres
        # creo una figura lo suficientemente grande 
        fig = plt.figure(figsize=(20,5))
        # pongo esto para que no se empalmen
        fig.tight_layout()
        
        num_clf = len(clasificadores) + 1
        num_ds = len(datasets)
        numeros = arange(num_clf,num_ds*num_clf+1, num_clf)
        
        # Iteramos sobre los datasets,labels y un identificador de las graficas de datos puros
        for dataset,label,id_ds in zip(datasets,labels,numeros):
            # id_ds es para ubicar las graficas de los datasets al final de cada renglon
            # tenemos un grid de 3 filas y 5 columnas
            print(num_ds,num_clf,id_ds)
            ax = plt.subplot(num_ds,num_clf,id_ds)
            # grafico al final de cada renglon el dataset original
            ax.scatter(dataset[:,0],dataset[:,1],c=label,cmap='plasma',s=20,alpha=1)
            ax.set_title("Dataset original")
            # Ahora iteramos por clasificadores, nombre y un identificador de clasificador
            for clasificador,nombre,id_clf in zip(clasificadores,nombres,range(1,num_clf+1)):
                # Entrenamos el clasificador
                clasificador.fit(dataset,label)
                # Realizamos la particion fina para graficar la frontera de desición
                Z,xx,yy = self.hacer_grid(dataset,clasificador)
                # Graficamos en el lugar deseado 
                ax = plt.subplot(num_ds,num_clf,(id_ds-num_clf)+id_clf)
                # Graficamos la partción fina (frontera de desición) 
                ax.pcolormesh(xx,yy,Z,cmap=ListedColormap(cmap_set1),alpha=0.2)
                # Graficamos alli mismo el dataset original 
                ax.scatter(dataset[:,0],dataset[:,1],c=label,cmap=ListedColormap(cmap_set1),s=20,alpha=1)
                # Definimos dominio y rango de la grafica
                ax.set_xlim(xx.min(),xx.max())
                ax.set_ylim(yy.min(),yy.max())
                # Agregamos la leyenda para identificar las clases 
                ax.legend(handles=patches)
                # Agregamos el titulo correspondiente a cada grafica
                ax.set_title(f"clasificador: {nombre} ")
                