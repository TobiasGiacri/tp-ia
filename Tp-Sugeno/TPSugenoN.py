import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import time
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

"""Subtractive Clustering Algorithm
"""


def SubstractiveClustering(data, Ra, Rb, AcceptRatio=0.3, RejectRatio=0.1):
    if Rb == 0:
        Rb = Ra * 1.15

    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)

    # 14/05/2020 cambio list comprehensions por distance matrix
    # P = np.array([np.sum([np.exp(-(np.linalg.norm(u-v)**2)/(Ra/2)**2) for v in ndata]) for u in ndata])
    # print(P)
    P = distance_matrix(ndata, ndata)
    alpha = (Ra / 2) ** 2
    P = np.sum(np.exp(-P ** 2 / alpha), axis=0)

    centers = []
    i = np.argmax(P)
    C = ndata[i]
    p = P[i]
    centers = [C]

    continuar = True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            P = P - p * np.array([np.exp(-np.linalg.norm(v - C) ** 2 / (Rb / 2) ** 2) for v in ndata])
        restarP = True
        i = np.argmax(P)
        C = ndata[i]
        p = P[i]
        if p > AcceptRatio * pAnt:
            centers = np.vstack((centers, C))
        elif p < RejectRatio * pAnt:
            continuar = False
        else:
            dr = np.min([np.linalg.norm(v - C) for v in centers])
            if dr / Ra + p / pAnt >= 1:
                centers = np.vstack((centers, C))
            else:
                P[i] = 0
                restarP = False
        if not any(v > 0 for v in P):
            continuar = False
    distancias = [[np.linalg.norm(p - c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(centers)
    return labels, centers

    """
Implementación similar a genfis2 de Matlab.
Sugeno type FIS. Generado a partir de clustering substractivo.
"""


def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean) ** 2.) / (2 * sigma ** 2.))


class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma


class fisInput:
    def __init__(self, min, max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids

    def view(self):
        x = np.linspace(self.minValue, self.maxValue, 20)
        # x = np.linspace(self.minValue -400,self.maxValue + 400,30)

        plt.figure()
        for m in self.centroids:
            s = (self.minValue - self.maxValue) / 8 ** 0.5
            y = gaussmf(x, m, s)
            plt.plot(x, y)


# Se genera la clase Fis (sugeno)
class fis:
    def __init__(self):
        self.rules = []
        self.memberfunc = []
        self.inputs = []
        self.solutions = []

    def genfis(self, data, labels, cluster_center):
        start_time = time.time()
        # labels, cluster_center = SubstactiveClustering(data, radii) se quita porque el clustering lo hacemos afuera para hacerlo variar a nuestro criterio

        n_clusters = len(cluster_center)
        cluster_center = cluster_center[:, :-1]

        varInput = data[:, :-1]

        # T = data[:,-1]
        maxValue = np.max(varInput, axis=0)
        minValue = np.min(varInput, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i], cluster_center[:, i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        varInput = data[:, :-1]  # hace esa anotacion fea porque porque queda mejor los datos presentados
        varOuput = data[:, -1]

        # ___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue - i.minValue) / np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(varInput, cluster, sigma), axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti, axis=1))
        varInput = np.c_[varInput, np.ones(len(varInput))]
        n_vars = varInput.shape[1]

        orden = np.tile(np.arange(0, n_vars), len(self.rules))
        acti = np.tile(nivel_acti, [1, n_vars])
        inp = varInput[:, orden]
        A = acti * inp / sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        solutions, residuals, rank, s = np.linalg.lstsq(A, varOuput, rcond=None)
        self.solutions = solutions
        print(solutions)
        return 0

    ###############################################################################################################

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

    #################################################################################################################

    def viewInputs(self):
        for input in self.inputs:
            input.view()

            # funcion de automatizacion


def automation(data_x, data_y, test_size, Ra,
               Rb):  # Parametros division de datos data,test_size; clustering substractivo Ra,Rb;
    '''                                 DIVISION DE LOS DATOS                              '''
    vda_entrenamiento, vda_esperados, tiempo_entrenamiento, tiempo_prueba = train_test_split(data_y, data_x,
                                                                                             test_size=test_size,
                                                                                             random_state=10)  # random_state modifica la distancia entre los puntos que tomo para test.
    dataTrain = np.hstack((tiempo_entrenamiento, vda_entrenamiento))  # datos de entrenamiento
    dataTest = np.hstack((tiempo_prueba, vda_esperados))  # datos testing
    '''                                 CLUSTERING SUSTRACTIVO                             '''
    labels, centers = SubstractiveClustering(dataTrain, Ra, Rb)
    plt.figure()
    plt.title(label='Clustering sustractivo para Ra={}'.format(Ra))
    plt.scatter(dataTrain[:, 0], dataTrain[:, 1], c=labels, s=7, label='Conjunto de prueba')
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=20, c="black", label='clusters')
    plt.legend()
    plt.show()
    '''                                 SUGENO                                             '''
    fis2 = fis()  # genero la clase fis, mejor llamado modelo de Sugeno
    fis2.genfis(dataTrain, labels, centers)
    fis2.viewInputs()

    '''                                 MODULO DE GRAFICO DE FUNCION SOLUCION DE SUGENO CON DATOS DE ENTRENAMIENTO                          '''

    maxValue = int(np.max(data_x))
    arreglo = np.linspace(1, maxValue,
                          maxValue)  # Generamos un arreglo con numeros del 1 al 350 para el eje x a graficar
    arreglo = arreglo.reshape(-1, 1)
    coordenadas_y = fis2.evalfis(arreglo)  # Generamos los valores del eje y a partir de la funcion solucion obtenida

    '''                                 MODULO DE VALIDACION Y CALCULO DE ERRORES                         datos_obtenidos '''
    errores = []
    resultado_obtenido = fis2.evalfis((tiempo_prueba))

    for i in range(len(resultado_obtenido)):
        resultado = resultado_obtenido[i] - vda_esperados[i]
        print(
            f" error en ierror{i}, error:  {resultado} , resultado_obtenido = {resultado_obtenido[i]}, resultado_esperado = {vda_esperados[i]}")
        errores.append(resultado)

    errores = np.sqrt(np.square(errores))
    promedio_error = np.mean(errores)

    plt.figure()
    plt.scatter(tiempo_entrenamiento, vda_entrenamiento, s=7, label='Conjunto de entrenamiento')
    plt.scatter(tiempo_prueba, vda_esperados, color='red', s=7, label='Conjunto de pruebas')
    plt.plot(arreglo, coordenadas_y, linestyle='-', color='yellow', label='Solucion sugeno')
    plt.xlabel('Tiempo')

    plt.legend()

    plt.title('Gráfico de Dispersión de Presión Arterial vs. Tiempo para Ra={}'.format(Ra))
    plt.grid(True)
    plt.show()

    print(f"Error promedio: {promedio_error}")
    return promedio_error * 100 / len(data_x)


'''                                 MODULO DE LECTURA DE ARCHIVO                                      '''

from sklearn.model_selection import train_test_split
import numpy as np

# Cargar los datos desde el archivo de texto
vda = np.loadtxt("samplesVDA2.txt")
cant_datos = vda.shape[0]
medidas_x_segundo = 400
npyarray = np.array
tiempo = np.arange(0, cant_datos / 400, 1 / medidas_x_segundo)
tiempo = tiempo * 1000

data_y = vda.reshape(-1, 1)
data_x = tiempo.reshape(-1, 1)

dataGral = np.hstack((data_y, data_x))

# Graficamos los datos totales
plt.scatter(data_x, data_y, s=7)
plt.xlabel('Tiempo (ms)')
plt.ylabel('VDA')
plt.title('Muestra Total')
plt.show()

errores_totales = []
vectorRa = []
Ra = 0
for i in range(0, 10):
    Ra += 0.1
    vectorRa.append(Ra)
    errores_totales.append(automation(data_x, data_y, 0.2, Ra, 0))
    print(errores_totales)

plt.figure()
plt.plot(vectorRa, errores_totales)
plt.ylabel('Errores promedio porcentual')
plt.xlabel('Ra (clustering substractivo)')
plt.title('Ra vs MSE')
plt.show()

ros = RandomOverSampler(random_state=10)

X_resampled, y_resampled = ros.fit_resample(data_x, data_y)

y_resampled2 = y_resampled.reshape(-1, 1)

print(y_resampled2.shape)

plt.figure()
plt.scatter(X_resampled, y_resampled2, s=7, c='blue', label='sintetico')

plt.xlabel('Tiempo (ms)')
plt.ylabel('VDA')
plt.title('Muestra Total')
plt.legend()
plt.show()

plt.figure()
plt.scatter(data_x, data_y, s=7, c='brown', label='data')
plt.legend()
plt.show()
