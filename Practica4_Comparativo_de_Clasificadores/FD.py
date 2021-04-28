"""
    Codigo tomado de: https://www.aprendemachinelearning.com/clasificar-con-k-nearest-neighbor-ejemplo-en-python/
    
    Se han modificado algunas lineas de codigo para adaptarlo al programa principal
    
"""

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from numpy import meshgrid,arange,c_
import matplotlib.patches as mpatches

class FronterasDeDesicion():
    def __init__(self, datasets,etiquetas,clasificadores,nombres):
        self.datasets = datasets
        self.labels = etiquetas
        self.clasificadores = clasificadores
        self.nombres = nombres


    def hacer_grid(self,X,y,clf):
        
        h = 0.02
        
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
    
    
    def mostrar(self):
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#c2f0c2']) 
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
        
        patch0 = mpatches.Patch(color='#FF0000', label='Clase 1')
        patch1 = mpatches.Patch(color='#00FF00', label='Clase 2')
        
        datasets = self.datasets
        labels = self.labels
        clasificadores = self.clasificadores
        nombres = self.nombres
        
        fig = plt.figure(figsize=(20,20))
        fig.tight_layout()
        
        for dataset,label,id_ds in zip(datasets,labels,[4,8,12]):
            ax = plt.subplot(3,4,id_ds)
            ax.scatter(dataset[:,0],dataset[:,1],c=label,cmap='plasma',s=2)
            ax.set_title("Dataset original")
            for clasificador,nombre,id_clf in zip(clasificadores,nombres,range(1,4)):
                clasificador.fit(dataset,label)
                Z,xx,yy = self.hacer_grid(dataset,label,clasificador)
                ax = plt.subplot(3,4,(id_ds-4)+id_clf)
                ax.pcolormesh(xx,yy,Z,cmap=cmap_light)
                ax.scatter(dataset[:,0],dataset[:,1],c=label,cmap = cmap_bold,s=2)
                ax.set_xlim(xx.min(),xx.max())
                ax.set_ylim(yy.min(),yy.max())
                ax.legend(handles=[patch0,patch1])
                ax.set_title(f"clasificador: {nombre} ")
                