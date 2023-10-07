import numpy as np
from matplotlib import pyplot as plt
import Clustering_Sustractivo
import GenFis


'''                                 MODULO DE LECTURA DE ARCHIVO                                      '''
vda   = np.loadtxt("/home/mikel/Documents/Facultad/IA/tp-ia/samplesVDA1.txt")                               #Lee el archivo

'''                                 MODULO DE FORMATEO DE ESTRUCTURAS                                 '''
cant_datos = vda.shape[0]                                           #Cantidad de filas/datos
medidas_x_segundo = 400                                             #Frecuencia de milisegundos
tiempo = np.arange(0,cant_datos/400,1/medidas_x_segundo)            #Arreglo intervalo tiempo en segundos
tiempo = tiempo*1000                                                #Arreglo intervalo tiempo en milisegundos

data_y = vda.reshape(-1,1)                                          #Prepara los datos, formatea de manera que haya una sola columna
data_x = tiempo.reshape(-1,1)                                       #Prepara los datos, Formatea de manera que haya una sola columna

'''                                 MODULO DE ELIMINACIÓN DE CASOS PARA TEST                          '''
ultimos_50_elementosvda =  data_y[-50:].copy()                      #Separa los ultimos 50 datos de entrada para test
data_y = data_y[:-50]

ultimos_50_elementostiempo = data_x[-50:].copy()                    #Separa los ultimos 50 datos de tiempo para test
data_x = data_x[:-50]
'''                                 MODULO DE COMBINACIÓN DE EJE X Y EJE Y EN UNA MATRIZ              '''
data = np.hstack((data_y,data_x))                                   #Arma la matriz

'''                                 MODULO DE EJECUCION Y GRAFICO DE CLUSTERING SUBSTRACTIVO          '''
sustractivo = Clustering_Sustractivo.Sustractivo()                                              #Crea instancia del algoritmo
r,c = sustractivo.subclust3(data,0.5,0)
plt.figure()
plt.scatter(data[:,1],data[:,0], c=r, s=7)
plt.scatter(c[:,1],c[:,0], marker='X',s=100,c="black")
plt.title('Clustering sustractivo')
plt.show()

'''                                 MODULO DE EJECUCION Y SOLUCION DE SUGENO                           '''


plt.figure()
plt.plot(data_x, data_y)
plt.show()



fis2 = GenFis.fis()                                                        #Crea instancia del algoritmo
print(data)
fis2.genfis(data, 0.5)                                              #A partir de los datos hace un clustering substractivo dentro (Si ya se es en vano haberlo hecho antes, aunque solo se le puede configurar el Ra)
fis2.viewInputs()                                                   #Calcula las funciones gaussianas y las muestra

'''                                 MODULO DE GRAFICO DE FUNCION SOLUCION DE SUGENO                           '''
arreglo = np.arange(1, 601, 2)                        #Generamos un arreglo con numeros del 1 al 600 para el eje x a graficar
arreglo = arreglo.reshape(-1,1)
r = fis2.evalfis(np.vstack(data_y))                                 #Generamos los valores del eje y a partir de la funcion solucion obtenida
plt.figure()
plt.plot(data_x,r,linestyle='--')
plt.title('Solucion sugeno')
plt.show()


'''                                 MODULO QUE HACE ALGO                                                       
fis2.solutions
r1 = data_x*-2.29539539+ -41.21850973
r2 = data_x*-15.47376916 -79.82911266
r3 = data_x*-15.47376916 -79.82911266
plt.plot(data_x,r1)
plt.plot(data_x,r2)
plt.plot(data_x,r3)
plt.title(' algox2')
plt.show()

'''