#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Parametric Entropic Locally Linear Embedding for Non-Linear Feature Extraction in Data Classification

"""

# Imports
import sys
import time
import warnings
import umap         # install with pip install umap_learn
import seaborn
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from numpy import trace
from numpy import dot
from numpy.linalg import det
from scipy.linalg import eigh
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
import sklearn.neighbors as sknn
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Calcula a divergência KL entre duas Gaussianas multivariadas
def divergenciaKL(mu1, mu2, cov1, cov2):
    m = len(mu1)
    
    # If covariance matrices are ill-conditioned
    if np.linalg.cond(cov1) > 1/sys.float_info.epsilon:
        cov1 = cov1 + np.diag(0.001*np.ones(m))
    if np.linalg.cond(cov2) > 1/sys.float_info.epsilon:
        cov2 = cov2 + np.diag(0.001*np.ones(m))
        
    dM1 = 0.5*(mu2-mu1).T.dot(inv(cov2)).dot(mu2-mu1)
    dM2 = 0.5*(mu1-mu2).T.dot(inv(cov1)).dot(mu1-mu2)
    dTr = 0.5*trace(dot(inv(cov1), cov2) + dot(inv(cov2), cov1))
    
    dKL = 0.5*(dTr + dM1 + dM2 - m)
    
    return dKL

# Simple PCA implementation
def myPCA(dados, d):

    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados


# Função que implementa o Local Linear Embedding (funciona OK)
def myLLE(X, num_neigh, d, alpha):
    # Gera o grafo KNN
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=num_neigh, mode='connectivity', include_self=False)
    A = knnGraph.toarray()  # Extrai a matriz de adjacência a partir do grafo KNN

    # Reconstruction weights matrix
    W = np.zeros(A.shape)

    # Step 1: estimate the optimum reconstruction weights
    um = np.ones(num_neigh)
    for i in range(A.shape[0]):
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = X[indices, :]
        C = np.zeros((num_neigh, num_neigh))
        for j in range(amostras.shape[0]):
            for k in range(amostras.shape[0]):
                #C[j, k] = np.dot((X[i, :] - amostras[j, :]), (X[i, :] - amostras[k, :]))
                C[j, k] = sum((X[i, :] - amostras[j, :])*(X[i, :] - amostras[k, :]))

        # Regularizes C
        D = np.eye(C.shape[0])
        C = C + alpha*D
        w = np.dot(np.linalg.inv(C), um)
        w = w/sum(w)
        W[i, indices] = w

    # Step 2: Find the embedding
    I = np.eye(W.shape[0])
    M = np.dot((I - W).T, (I - W))
    lambdas, alphas = eigh(M, eigvals=(1, d))   # descarta menor autovalor (zero)
    
    return alphas

# Parametric Entropic LLE
def EntropicLLE(X, num_neigh, d, alpha, mode):
    # Computa a média e a matriz de covariâncias para cada patch
    medias = np.zeros((dados.shape[0], dados.shape[1]))
    matriz_covariancias = np.zeros((dados.shape[0], dados.shape[1], dados.shape[1]))

    # Gera o grafo KNN
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=num_neigh, mode='connectivity', include_self=False)
    A = knnGraph.toarray()  # Extrai a matriz de adjacência a partir do grafo KNN

    for i in range(X.shape[0]):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # pontos isolados recebem média igual a zero
            medias[i, :] = amostras[i, :]
            matriz_covariancias[i, :, :] = np.eye(dados.shape[1])
        else:
            amostras = dados[indices]
            medias[i, :] = amostras.mean(0)
            matriz_covariancias[i, :, :] = np.cov(amostras.T)

    # Reconstruction weights matrix
    W = np.zeros(A.shape)

    # Step 1: estimate the optimum reconstruction weights
    um = np.ones(num_neigh)
    for i in range(A.shape[0]):
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = X[indices, :].copy()
        C = np.zeros((num_neigh, num_neigh))
        for j in range(amostras.shape[0]):
            for k in range(amostras.shape[0]):                
                Dij = divergenciaKL(medias[i, :], medias[j, :], matriz_covariancias[i, :, :], matriz_covariancias[j, :, :])
                Dik = divergenciaKL(medias[i, :], medias[k, :], matriz_covariancias[i, :, :], matriz_covariancias[k, :, :])
                if mode == 'product':
                    C[j, k] = Dij*Dik
                else:
                    C[j, k] = abs(Dij - Dik)

        # Regularizes C
        D = np.eye(C.shape[0])
        C = C + alpha*D
        w = np.dot(np.linalg.inv(C), um)
        w = w/sum(w)
        W[i, indices] = w

    # Step 2: Find the embedding
    I = np.eye(W.shape[0])
    M = np.dot((I - W).T, (I - W))
    lambdas, alphas = eigh(M, eigvals=(1, d))   
    
    return alphas    


# Modified version (with class labels)
def FastEntropicLLE(X, num_neigh, d, alpha, target, modo):
    # Computa a média e a matriz de covariâncias para cada patch
    medias = np.zeros((X.shape[0], X.shape[1]))
    matriz_covariancias = np.zeros((X.shape[0], X.shape[1], X.shape[1]))

    # Gera o grafo KNN
    knnGraph = sknn.kneighbors_graph(X, n_neighbors=num_neigh, mode='connectivity', include_self=False)
    A = knnGraph.toarray()  # Extrai a matriz de adjacência a partir do grafo KNN

    if modo == 'supervised':
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # Se for de classes diferentes, desconecta a aresta
                if A[i, j] > 0 and target[i] != target[j]:
                    A[i, j] = 0

    for i in range(X.shape[0]):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # pontos isolados recebem média igual a zero
            medias[i, :] = X[i, :]
            matriz_covariancias[i, :, :] = np.eye(X.shape[1])
        else:
            amostras = X[indices]
            medias[i, :] = X.mean(0)
            matriz_covariancias[i, :, :] = np.cov(X.T)

    # Reconstruction weights matrix
    W = np.zeros(A.shape)

    # Step 1: estimate the optimum reconstruction weights
    um = np.ones(num_neigh)
    for i in range(A.shape[0]):
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        amostras = X[indices, :].copy()
        C = np.zeros((num_neigh, num_neigh))
        vetor_dij = np.zeros(num_neigh)
        for j in range(amostras.shape[0]):
            vetor_dij[j] = divergenciaKL(medias[i, :], medias[j, :], matriz_covariancias[i, :, :], matriz_covariancias[j, :, :])
        C = np.outer(vetor_dij, vetor_dij)
            
        # Regularizes C
        D = np.eye(C.shape[0])
        C = C + alpha*D

        w = np.dot(np.linalg.inv(C), um)
        w = w/sum(w)

        #W[i, indices] = w  # Esse era o original
        r = 0
        for j in indices:
            W[i, j] = w[r]
            r += 1

    # Step 2: Find the embedding
    I = np.eye(W.shape[0])
    M = np.dot((I - W).T, (I - W))
    lambdas, alphas = eigh(M, eigvals=(1, d))   
    
    return alphas    


'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    print('KNN accuracy: ', acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)
    print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    print('QDA accuracy: ', acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)
    print('RFC accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    print('Average accuracy: ', average)
    print('Maximum accuracy: ', maximo)
    print()

    return [sc, average]


# Usa estratégia de busca para o melhor valor de K na construção do grafo KNN
def batch_LLEKL_KNN(dados, target):

    n = dados.shape[0]

    # Search for the best K
    inicio = 2
    incremento = 1
    fim = min(n//2, 41)    # se fim = n testa todos os tamanhos de patch possíveis
    vizinhos = list(range(inicio, fim, incremento))
    acuracias = []
    scs = []

    for viz in vizinhos:
        print('K = %d' %viz)
        dados_lle_ent = FastEntropicLLE(X=dados, num_neigh=viz, d=2, alpha=0.001, target=target, modo='unsupervised') 
        s = 'PELLE'
        L_llekl = Classification(dados_lle_ent.T, target, s)
        scs.append(L_llekl[0])
        acuracias.append(L_llekl[1])

    print('List of values for K: ', vizinhos)
    print('Supervised classification accuracies: ', acuracias)
    acuracias = np.array(acuracias)
    print('Best Acc: ', acuracias.max())
    print('K* = ', vizinhos[acuracias.argmax()])
    print()

    plt.figure(1)
    plt.plot(vizinhos, acuracias)
    plt.title('Mean accuracies for different values of K')
    plt.show()

    print('List of values for K: ', vizinhos)
    print('Silhouette Coefficients: ', scs)
    scs = np.array(scs)
    print('Best SC: ', scs.max())
    print('K* = ', vizinhos[scs.argmax()])
    print()

    plt.figure(2)
    plt.plot(vizinhos, scs, color='red')
    plt.title('Silhouette coefficients for different values of K')
    plt.show()


#%%%%%%%%%%%%%%%%%%%%  Data loading

# Scikit-learn datasets

X = skdata.load_wine() 
#X = skdata.fetch_openml(name='prnn_crabs', version=1)
#X = skdata.fetch_openml(name='tae', version=1) 
#X = skdata.fetch_openml(name='hayes-roth', version=2) 
#X = skdata.fetch_openml(name='plasma_retinol', version=2)
#X = skdata.fetch_openml(name='parity5', version=1) 
#X = skdata.fetch_openml(name='thoracic_surgery', version=1) 
#X = skdata.fetch_openml(name='conference_attendance', version=1)
#X = skdata.fetch_openml(name='tic-tac-toe', version=1) 
#X = skdata.fetch_openml(name='sa-heart', version=1) 
#X = skdata.fetch_openml(name='aids', version=1)
#X = skdata.fetch_openml(name='haberman', version=1)
#X = skdata.fetch_openml(name='breast-tissue', version=2)
#X = skdata.fetch_openml(name='mu284', version=2)  
#X = skdata.fetch_openml(name='analcatdata_wildcat', version=2)
#X = skdata.fetch_openml(name='ar1', version=1) 
#X = skdata.fetch_openml(name='chscase_geyser1', version=2)
#X = skdata.fetch_openml(name='bolts', version=2)
#X = skdata.fetch_openml(name='lupus', version=1)
#X = skdata.fetch_openml(name='monks-problems-1', version=1)
#X = skdata.fetch_openml(name='corral', version=1)
#X = skdata.fetch_openml(name='acute-inflammations', version=2) 
#X = skdata.fetch_openml(name='visualizing_environmental', version=2)  
#X = skdata.fetch_openml(name='vineyard', version=2) 
#X = skdata.fetch_openml(name='kidney', version=2)      


dados = X['data']
target = X['target']  

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

print('############## Parametric Entropic LLE ####################')
batch_LLEKL_KNN(dados, target)   # Comentar essa linha se for usar grafo KNN

#%%%%%%%%%%%% Simple PCA
dados_pca = myPCA(dados, 2)

#%%%%%%%%%%% ISOMAP
model = Isomap(n_neighbors=20, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%% LLE
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#%%%%%%%%%%% Hessian LLE
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='hessian', eigen_solver='dense')
#model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='hessian')
dados_LLE_h = model.fit_transform(dados)
dados_LLE_h = dados_LLE_h.T

#%%%%%%%%%%% LTSA
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='ltsa', eigen_solver='dense')
#model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='ltsa')
dados_LLE_l = model.fit_transform(dados)
dados_LLE_l = dados_LLE_l.T

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados)
dados_umap = dados_umap.T

#%%%%%%%%% Classifica dados
L_pca = Classification(dados_pca, target, 'PCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lle_h = Classification(dados_LLE_h, target, 'Hessian LLE')
L_lle_l = Classification(dados_LLE_l, target, 'LTSA')
L_umap = Classification(dados_umap, target, 'UMAP')
