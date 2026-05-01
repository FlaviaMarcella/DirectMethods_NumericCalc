"""
Módulo: pagerank.py
Descrição: Implementa o método de cálculo do algoritmo PageRank através
da formulação de sistema linear e resolução via métodos diretos.
"""

import numpy as np
from gauss import resolver_gauss

def calcular_pagerank(G, alpha=0.85):
    """
    Calcula o PageRank dado a matriz de adjacencia G.
    
    Parametros:
        G: Matriz de adjacencia (quadrada)
        alpha: Fator de amortecimento (padrao: 0.85)
        
    Retorna:
        Vetor normalizado de ranks pi
    """
    n = G.shape[0]
    G = np.array(G, dtype=float)
    
    # Evita divisao por zero para paginas sem links (dangling nodes)
    somas_coluna = G.sum(axis=0)
    somas_coluna[somas_coluna == 0] = 1
    
    # Constroi a matriz de transicao P
    P = G / somas_coluna
    
    # Sistema: (I - alpha*P.T) * pi = (1-alpha)/n * e
    I = np.eye(n)
    e = np.ones(n)
    
    A_page = I - alpha * P.T
    b_page = ((1 - alpha) / n) * e
    
    # Resolve utilizando Gauss com Pivoteamento Parcial
    pi = resolver_gauss(A_page, b_page)
    
    return pi