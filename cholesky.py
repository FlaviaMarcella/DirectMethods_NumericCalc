"""
Módulo: cholesky.py
Descrição: Implementa a Fatoração de Cholesky (A = L * L.T), exigida para 
matrizes Simétricas e Positivas Definidas (SPD). Ideal para Mínimos Quadrados.
"""

import numpy as np

def subst_retro(U, y):
    """ Substituicao retroativa: resolve Ux = y. """
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def subst_prog(L, b):
    """ Substituicao progressiva: resolve Ly = b. """
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def cholesky(A):
    """ 
    Fatoracao de Cholesky: A = L @ L.T
    Levanta ValueError se a matriz nao for SPD.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))

    for k in range(n):
        soma_diag = A[k, k] - np.sum(L[k, :k] ** 2)
        if soma_diag <= 0:
            raise ValueError(f"Matriz nao SPD: elemento diagonal {k} negativo ou nulo.")
        L[k, k] = np.sqrt(soma_diag)
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - np.sum(L[i, :k] * L[k, :k])) / L[k, k]
    return L

def resolver_cholesky(A, b):
    """ Resolve o sistema Ax = b via Fatoracao de Cholesky. """
    L = cholesky(A)
    y = subst_prog(L, b)
    # L.T eh uma matriz triangular superior
    x = subst_retro(L.T, y)
    return x, L