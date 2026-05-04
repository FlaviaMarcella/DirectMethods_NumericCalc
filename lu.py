"""
Módulo: lu.py
Descrição: Implementa o método de Fatoração LU de Doolittle (sem pivoteamento)
e as substituições progressiva e retroativa para a resolução otimizada de 
sistemas com múltiplos vetores de carga. Inclui também o LU Vetorizado (Desafio).
"""

import numpy as np
import time

def subst_retro(U, y):
    """ Substituicao retroativa: resolve Ux = y. """
    n = len(y)
    x = np.zeros(n, dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Divisao por zero na substituicao retroativa.")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def subst_prog(L, b):
    """ Substituicao progressiva: resolve Ly = b. """
    n = len(b)
    y = np.zeros(n, dtype=L.dtype)
    for i in range(n):
        if L[i, i] == 0:
            raise ValueError("Divisao por zero na substituicao progressiva.")
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def fatoracao_lu(A):
    """ Fatoracao de Doolittle: A = LU sem pivoteamento. """
    tipo = A.dtype
    n = A.shape[0]
    L = np.eye(n, dtype=tipo)
    U = np.zeros((n, n), dtype=tipo)

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

def experimento_desempenho_lu(n_c=200, n_b=50):
    """
    Compara o tempo de execução entre Gauss repetido e LU reutilizado.
    """
    from gauss import resolver_gauss
    
    # Garantir reprodutibilidade local
    np.random.seed(42)
    
    Ac = np.random.rand(n_c, n_c).astype(np.float32) + np.eye(n_c) * n_c
    bs = [np.random.rand(n_c).astype(np.float32) for _ in range(n_b)]

    # Gauss Repetido
    t0 = time.time()
    for bi in bs: 
        resolver_gauss(Ac, bi)
    t_gauss_total = time.time() - t0

    # LU Reutilizado
    t0 = time.time()
    L, U = fatoracao_lu(Ac)
    for bi in bs: 
        y = subst_prog(L, bi)
        x = subst_retro(U, y)
    t_lu_total = time.time() - t0
    
    return t_gauss_total, t_lu_total

def fatoracao_lu_vectorized(A):
    """ Fatoracao LU otimizada com operacoes vetorizadas do NumPy (Produto Externo). """
    n = A.shape[0]
    tipo = A.dtype
    U = A.copy()
    L = np.eye(n, dtype=tipo)

    for k in range(n - 1):
        if U[k, k] == 0:
            raise ValueError("Pivo nulo na fatoracao vetorizada.")
        
        L[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k:] -= np.outer(L[k+1:, k], U[k, k:])
        
    return L, U
