"""
Módulo: pagerank.py
Descrição: Implementa o algoritmo PageRank utilizando a resolução de 
sistemas lineares (Método Direto).
"""

import numpy as np
from gauss import resolver_gauss

def calcular_pagerank(G, alpha=0.85):
    """
    Calcula o vetor de PageRank pi para uma matriz de adjacência G.
    G[i, j] = 1 se existe link da página j para a página i.
    """
    n = G.shape[0]
    G = G.astype(float)
    
    # 1. Lidar com nós sem saída (dangling nodes)
    # Se uma coluna é toda zero, a página não tem links. 
    # Distribuímos a probabilidade igualmente para todas as páginas.
    somas_coluna = np.sum(G, axis=0)
    for j in range(n):
        if somas_coluna[j] == 0:
            G[:, j] = 1.0 / n
        else:
            G[:, j] /= somas_coluna[j]
            
    # G agora é a matriz de transição P (estocástica por colunas)
    P = G
    
    # 2. Montar o sistema linear (I - alpha*P) pi = (1-alpha)/n * e
    # Onde e é um vetor de uns.
    I = np.eye(n)
    A = I - alpha * P
    b = np.ones(n) * (1 - alpha) / n
    
    # 3. Resolver o sistema
    pi = resolver_gauss(A, b)
    
    # Normalizar (garantir que a soma seja 1)
    pi = pi / np.sum(pi)
    
    return pi
