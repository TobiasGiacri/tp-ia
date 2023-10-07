
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import Clustering_Sustractivo
import time

'''                                 CLASE CON EL ALGORITMO DE SUGENO                                      '''
#                                   Funcion de calculo de funciones gaussianas
# data: Arreglo de datos
# mean: arreglo de centros
# sigma: desviacion estandar de los centros
def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean) ** 2.) / (2 * sigma ** 2.))


class fisRule:
    #                                   Funcion de iniciacion/constructor de fisRul?
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput: #lase dedicada a lo grafico
    #                                   Funcion de iniciacion/constructor de fisInput
    def __init__(self, min, max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids

    #                                   Funcion de mostrar las funciones gaussianas
    def view(self):
        # x = np.linspace(self.minValue,self.maxValue,20)
        x = np.linspace(self.minValue - 400, self.maxValue + 400, 30)

        plt.figure()
        for m in self.centroids:
            s = (self.minValue - self.maxValue) / 8 ** 0.5
            y = gaussmf(x, m, s)
            plt.plot(x, y)

        plt.title('Funciones gaussianas')
        plt.show()

class fis:
    def __init__(self):
        self.rules = []
        self.memberfunc = []
        self.inputs = []

    #                                   Funcion de generacion de solucion de sugeno
    # data: Arreglo de datos a entrar
    # radii: radio intercluster del sustractivo
    # devuelve:

    def genfis(self, data, radii):
        start_time = time.time() # El método comienza midiendo el tiempo de ejecución.
        sustractivo = Clustering_Sustractivo.Sustractivo()
        labels, cluster_center = sustractivo.subclust3(data, radii, 0)
        n_clusters = len(cluster_center)
        cluster_center = cluster_center[:, :-1]                                        #se recorta las otras columnas y me quedo solo con la primera columna (coordenas y de los centros de cluster

        P = data[:, :-1]   # Se obtienen las coordenadas en y de los puntos en cuestion,

        maxValue = np.max(P, axis=0)#  se calculan los valores máximos (maxValue) y mínimos (minValue) de cada variable de entrada.
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i], cluster_center[:, i]) for i in range(len(maxValue))]
        #  Se crea una lista de objetos fisInput para cada variable de entrada. Cada objeto fisInput se inicializa con su valor máximo, mínimo y
        #  los centroides de los clusters correspondientes a esa variable.
        self.rules = cluster_center
        self.entrenar(data)

    ######################################################################################################################
    def entrenar(self, data):
        P = data[:, :-1]
        T = data[:, -1]
        # P contiene las variables de entrada del conjunto de datos data.
        # T contiene las salidas deseadas (target) del conjunto de datos data.

        # ___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue - i.minValue) / np.sqrt(8) for i in self.inputs])
        # Aquí se calcula la desviación estándar sigma para las funciones de membresía gaussianas utilizadas en el sistema Sugeno.
        #  self.inputs contiene información sobre las variables de entrada.

        f = [np.prod(gaussmf(P, cluster, sigma), axis=1) for cluster in self.rules]
        #  Se calcula la activación de cada regla para cada instancia del conjunto de datos P.
        #  Esto se hace multiplicando las funciones de membresía gaussianas por cada regla y luego tomando el producto.
        #  El resultado es una lista de valores de activación para cada regla y cada instancia de entrada.

        nivel_acti = np.array(f).T
        #  nivel_acti es una matriz donde cada fila corresponde a una instancia del conjunto de datos y cada columna corresponde a una regla.
        #  Contiene los valores de activación de cada regla para cada instancia.

        # print("nivel acti")
        # print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti, axis=1))
        # sumMu es un vector que contiene la suma de los valores de activación para cada instancia.
        # Esto se usa en el cálculo de los coeficientes del sistema Sugeno.
        # print("sumMu")
        # print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0, n_vars), len(self.rules))
        acti = np.tile(nivel_acti, [1, n_vars])
        inp = P[:, orden]
        # Aquí se preparan los datos para el cálculo de los coeficientes. Se añade una columna de unos a la matriz P,
        # luego se calculan las matrices acti e inp que se utilizan para construir la matriz de coeficientes A.

        A = acti * inp / sumMu
        # A es la matriz de coeficientes que se utiliza en el sistema de ecuaciones lineales.

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        self.solutions = solutions  # .reshape(n_clusters,n_vars)
        # Se resuelve el sistema de ecuaciones lineales utilizando mínimos cuadrados y se almacenan las soluciones en self.solutions.
        # print(solutions)
        return 0


    #  Metodo que evalua los puntos de entrada en la funcion solucion obtenida por sugeno
    #Precondicion: Solucion encontrada antes de entrar al metodo
    #data: arreglo de puntos en eje x a graficar
    #devuelve: Arreglo con las coordenadas en y correspondientes a los valores de entrada
    def evalfis(self, data):
        sigma = np.array([(input.maxValue - input.minValue) for input in self.inputs]) / np.sqrt(8)
        f = [np.prod(gaussmf(data, cluster, sigma), axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti, axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0, n_vars), n_clusters)
        acti = np.tile(nivel_acti, [1, n_vars])
        inp = P[:, orden]
        coef = self.solutions
        return np.sum(acti * inp * coef / sumMu, axis=1)

    #Metodo que se encarga de graficar las funciones gaussianas obtenidas por sugeno
    def viewInputs(self):
        for input in self.inputs:
            input.view()

