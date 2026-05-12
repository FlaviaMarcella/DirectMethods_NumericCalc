"""
Módulo: lu.py
Descrição: Implementa o método de Fatoração LU de Doolittle (sem pivoteamento)
e as substituições progressiva e retroativa. Otimizado com vetorização.
"""

import numpy as np
import time

def fatoracao_lu(A):
    """ Fatoração de Doolittle: A = LU sem pivoteamento .
    Retorna (L, U).
    """
    A = np.array(A, dtype=np.float32)
    n = A.shape[0]
    L = np.eye(n, dtype=np.float32)
    U = np.zeros((n, n), dtype=np.float32)

    for k in range(n):
        # Linha k de U
        for j in range(k, n):
            U[k, j] = A[k, j] - L[k, :k] @ U[:k, j]
        # Coluna k de L
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Divisão por zero: Matriz singular ou pivô zero detectado durante a fatoração LU.")
            L[i, k] = (A[i, k] - L[i, :k] @ U[:k, k]) / U[k, k]

    return L, U


def subst_prog(L, b):
    """ Substituição progressiva: resolve Ly = b.
    """
    n = len(b)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if L[i, i] == 0:
            raise ValueError("Divisão por zero: Matriz L singular ou pivô zero detectado na substituição progressiva.")
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def subst_retro(U, y):
    """ Substituição retroativa: resolve Ux = y.
    """
    n = len(y)
    x = np.zeros(n, dtype=np.float32)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Divisão por zero: Matriz U singular ou pivô zero detectado na substituição retroativa.")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def fatoracao_lu_vectorized(A):
    """ Fatoração de Doolittle: A = LU com vetorização NumPy.
    Versão otimizada usando operações vetorizadas.
    Retorna (L, U).
    """
    A = np.array(A, dtype=np.float32)
    n = A.shape[0]
    L = np.eye(n, dtype=np.float32)
    U = np.zeros((n, n), dtype=np.float32)

    for k in range(n):
        # Linha k de U (vetorizado)
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        
        # Coluna k de L (vetorizado)
        if U[k, k] == 0:
            raise ValueError("Divisão por zero: Matriz singular ou pivô zero detectado durante a fatoração LU.")
        L[k+1:, k] = (A[k+1:, k] - L[k+1:, :k] @ U[:k, k]) / U[k, k]

    return L, U


def resolver_lu(A, b):
    """ Resolve Ax = b via fatoração LU.
    """
    L, U = fatoracao_lu(A)
    y = subst_prog(L, b)
    x = subst_retro(U, y)
    return x, L, U