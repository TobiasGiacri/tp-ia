
'''                                 CLASE CON ALGORITMO DE CLUSTERING SUBSTRACTIVO              '''
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix

#data: arreglo de datos de entrada de dos columnas
#Ra: radio intracluster
#Rb: radio intercluster
#AcceptRatio: Ratio de aceptacion
#RejectRatio: Ratio de rechazo
#Devuelve: labels: arreglo de pertenencia a clusters    centers: arreglo de coordenadas de clusters encontrados
class Sustractivo:
    def subclust3(self, data, Ra, Rb, AcceptRatio=0.3, RejectRatio=0.1):
        if Rb==0:
            Rb = Ra*1.15
        scaler = MinMaxScaler()
        scaler.fit(data)
        ndata = scaler.transform(data)

        P = distance_matrix(ndata,ndata)
        alpha=(Ra/2)**2
        P = np.sum(np.exp(-P**2/alpha),axis=0)

        i=np.argmax(P)
        C = ndata[i]
        p=P[i]
        centers = [C]

        continuar=True
        restarP = True
        while continuar:                                               #Ciclo iterativo para encontrar centroides
            pAnt = p
            if restarP:
                P=P-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in ndata])
            restarP = True
            i=np.argmax(P)
            C = ndata[i]
            p=P[i]
            if p>AcceptRatio*pAnt:
                centers = np.vstack((centers,C))
            elif p<RejectRatio*pAnt:
                continuar=False
            else:
                dr = np.min([np.linalg.norm(v-C) for v in centers])
                if dr/Ra+p/pAnt>=1:
                    centers = np.vstack((centers,C))
                else:
                    P[i]=0
                    restarP = False
            if not any(v>0 for v in P):
                continuar = False
        distancias = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
        labels = np.argmin(distancias, axis=0)
        centers = scaler.inverse_transform(centers)
        return labels, centers
