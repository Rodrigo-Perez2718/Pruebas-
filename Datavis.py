class Astrodata:
    """
    Astrodata es una librería para ciencias de datos en astronomía. Proporciona varias funciones para agrupación y visualización de datos.

    Parámetros:
        x (array-like): Datos de entrada que representan los valores del eje x, serán transformados a una matriz numpy.
        y (array-like): Datos de entrada que representan los valores del eje y, serán transformados a una matriz numpy.

    Métodos:
        DBSCAn(self, eps, title, min_samples, x_label, y_label):
            Realiza agrupación espacial basada en densidad de aplicaciones con ruido (DBSCAN) en los puntos de datos y visualiza los resultados utilizando un gráfico de dispersión.

            Parámetros:
                eps (float): La distancia máxima que define el vecindario de un punto.
                title (str): Título del gráfico de dispersión.
                min_samples (int): El número mínimo de muestras requeridas para que un punto sea considerado un punto central.
                x_label (str): Etiqueta para el eje x del gráfico de dispersión.
                y_label (str): Etiqueta para el eje y del gráfico de dispersión.


        Kme(self, n_clusters, title, x_label, y_label, show_centroids=True):
            Realiza agrupación K-means en los puntos de datos y visualiza los resultados utilizando un gráfico de dispersión.

            Parámetros:
                n_clusters (int): Número de grupos o clusters en los que se agruparán los datos.
                title (str): Título del gráfico de dispersión.
                x_label (str): Etiqueta para el eje x del gráfico de dispersión.
                y_label (str): Etiqueta para el eje y del gráfico de dispersión.
                show_centroids (bool, opcional): Si es True, se mostrarán los centroides de los clusters en el gráfico. Por defecto es True.

        dendrogram(self, method):
            Genera un dendrograma para los puntos de datos.

            Parámetros:
                method (str): El método de vinculación para la agrupación jerárquica; por ejemplo, "ward".

    Nota: La clase `Astrodata` requiere las siguientes librerías de Python: numpy, sklearn, matplotlib y scipy.
    """
    def __init__(self, x,y):
        #Definiremos las variables x e y, las cuales se transformarán a matrices respectivamente.
        import numpy as np
        self.x = x
        self.y = y
        self.x = np.array(self.x)
        self.y = np.array(self.y)
    def DBSCAn(self, eps, title, min_samples, x_label, y_label):
        
        #Siendo eps= ε,que corresponde a la distancia máxima que define el vecindario de un punto.
        from sklearn.cluster import DBSCAN
        import matplotlib.pyplot as plt
        import numpy as np
        self.eps = eps
        self.min_samples = min_samples
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan_labels = dbscan.fit_predict(np.column_stack((self.x, self.y)))
        #Visualización de los resultados de la agrupación realizada por el algoritmo DBSCAN.
        plt.figure(figsize=(10,6))
        plt.scatter(self.x, self.y, c=dbscan_labels, cmap='viridis')
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()
    def Kme(self, n_clusters, title, x_label, y_label, show_centroids=True):
        # Importa las librerias necesarias para su ejecución
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt 
        import numpy as np
        # Ejecuta un clustering tipo K-means con gráfica tipo scatter plot.
        # Esta función utiliza el algoritmo K-means para agrupar los datos en `n_clusters` grupos y muestra
        # un scatter plot de los datos originales junto con los puntos coloreados según su cluster.

        # Parámetros:
        #    - n_clusters (int): Número de clusters o grupos en los que se agruparán los datos.
        #    - title (str): Título del gráfico.
        #    - x_label (str): Etiqueta del eje x del gráfico.
        #    - y_label (str): Etiqueta del eje y del gráfico.
        #    - show_centroids (bool, opcional): Si es True, muestra los centroides de los clusters en el gráfico. Por defecto es True.

        # Se almacenan los parámetros pasados como atributos de la clase para poder reutilizarlos en otras funciones
        self.n_clusters = n_clusters
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        # Se crea un modelo de K-means con el número de clusters especificado
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)

        # Se ajusta el modelo a los datos
        # Se utiliza np.column_stack para apilar verticalmente los arrays de x e y, ya que K-means requiere una matriz bidimensional
        kmeans.fit(np.column_stack((self.x, self.y)))
        # Se obtienen las coordenadas de los centroides y las etiquetas de los clusters para cada punto
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Se inicia la creación del gráfico
        plt.figure(figsize=(10, 6))

        # Plotting the original data
        plt.subplot(2, 1, 1)
        plt.scatter(self.x,self.y,  label='_nolegend_')
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid()
        
         # Scatter plot de los datos con colores según las etiquetas de los clusters
        plt.subplot(2, 1, 2)
        plt.scatter(self.x, self.y, c=labels, cmap='viridis', label='Datos')  # Agregamos una etiqueta 'Datos' para que aparezca en la leyenda
        if show_centroids:
            # Si show_centroids es True, se muestran los centroides en el gráfico
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200, label='Centroides')  # Agregamos una etiqueta 'Centroides' para que aparezca en la leyenda
        plt.title(self.title + ' - Clustering')  # Título del gráfico
        plt.xlabel(self.x_label)  # Etiqueta del eje x
        plt.ylabel(self.y_label)  # Etiqueta del eje y
        plt.legend()  # Mostrar leyenda con las etiquetas proporcionadas
        plt.tight_layout()  # Ajustar el diseño del gráfico para evitar superposición de elementos
        plt.grid()  # Mostrar cuadrícula
        plt.show()  # Mostrar el gráfico completo
    def dendrogram(self, method):
        # Genera un dendrograma para los datos dados
        # Se requiere insertar el método de la forma "Method" ; ejemplo: "ward"
        from scipy.cluster.hierarchy import dendrogram, linkage
        import matplotlib.pyplot as plt 
        import numpy as np
        X = np.column_stack((self.x, self.y))
        Z = linkage(X, method=method)
        plt.figure(figsize=(10, 10))
        dendrogram(Z)
        plt.show()
        