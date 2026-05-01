"""
Módulo: lu.py
Descrição: Implementa o método de Fatoração LU de Doolittle (sem pivoteamento)
e as substituições progressiva e retroativa para a resolução otimizada de 
sistemas com múltiplos vetores de carga. Inclui também o LU Vetorizado (Desafio).
"""

import numpy as np

def subst_retro(U, y):
    """ Substituicao retroativa: resolve Ux = y. """
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Divisao por zero na substituicao retroativa.")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def subst_prog(L, b):
    """ Substituicao progressiva: resolve Ly = b. """
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        if L[i, i] == 0:
            raise ValueError("Divisao por zero na substituicao progressiva.")
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def fatoracao_lu(A):
    """ Fatoracao de Doolittle: A = LU sem pivoteamento. """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - L[k, :k] @ U[:k, j]
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Divisao por zero na Fatoracao LU.")
            L[i, k] = (A[i, k] - L[i, :k] @ U[:k, k]) / U[k, k]
    return L, U

def resolver_lu(A, b):
    """ Resolve Ax = b utilizando a fatoracao LU. """
    L, U = fatoracao_lu(A)
    y = subst_prog(L, b)
    x = subst_retro(U, y)
    return x, L, U

def fatoracao_lu_vectorized(A):
    """ Fatoracao LU otimizada com operacoes vetorizadas do NumPy (Produto Externo). """
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)

    for k in range(n - 1):
        if U[k, k] == 0:
            raise ValueError("Pivo nulo na fatoracao vetorizada.")
        
        # Otimizacao via fatiamento e produto externo
        L[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k:] -= np.outer(L[k+1:, k], U[k, k:])
        
    return L, U