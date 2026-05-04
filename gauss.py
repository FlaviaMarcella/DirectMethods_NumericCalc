"""
Módulo: gauss.py
Descrição: Implementa a Eliminação de Gauss clássica, a Eliminação de Gauss 
com pivoteamento parcial e uma versão com detecção de singularidade.
Força rigorosamente as precisões de float32 quando solicitadas.
"""

import numpy as np

def subst_retro(U, y):
    """ Substituicao retroativa: resolve Ux = y para matrizes triangulares superiores. """
    n = len(y)
    x = np.zeros(n, dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Divisao por zero na substituicao retroativa.")
        # Garante que a operação respeite o dtype original (float32 ou float64)
        soma = U[i, i + 1:] @ x[i + 1:]
        x[i] = (y[i] - soma) / U[i, i]
    return x

def gauss_sem_pivoteamento(A, b):
    """ Eliminacao de Gauss SEM pivoteamento. """
    tipo = A.dtype
    n = len(b)
    A = A.copy()
    b = b.copy()
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i]     -= m * b[k]
    return A, b

def resolver_gauss_sem(A, b):
    """ Resolve Ax = b usando Gauss sem pivoteamento. """
    Au, bu = gauss_sem_pivoteamento(A, b)
    return subst_retro(Au, bu)

def gauss(A, b):
    """ Eliminacao de Gauss com pivoteamento parcial. """
    tipo = A.dtype
    n = len(b)
    A = A.copy()
    b = b.copy()
    for k in range(n - 1):
        # Pivoteamento parcial
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i]     -= m * b[k]
    return A, b

def resolver_gauss(A, b):
    """ Resolve Ax = b usando Gauss com pivoteamento parcial. """
    Au, bu = gauss(A, b)
    return subst_retro(Au, bu)

def experimento_estabilidade_gauss(a11_vals):
    """
    Executa o experimento de estabilidade (Q1.4) variando a11.
    Retorna os erros relativos para as versões sem e com pivoteamento.
    """
    erro_sem, erro_com = [], []
    x_true = np.array([1/3, 2/3], dtype=np.float32)

    for a11 in a11_vals:
        A_p = np.array([[a11, 3], [1, 1]], dtype=np.float32)
        b_p = np.array([2.0001, 1], dtype=np.float32)
        
        xs = resolver_gauss_sem(A_p, b_p)
        xc = resolver_gauss(A_p, b_p)
        
        erro_sem.append(abs(xs[0] - x_true[0]) / x_true[0])
        erro_com.append(abs(xc[0] - x_true[0]) / x_true[0])
        
    return erro_sem, erro_com

LIMIAR_SINGULARIDADE = 1e-12

def gauss_singular(A, b):
    """ Eliminacao de Gauss com deteccao rigorosa de quase-singularidade. """
    tipo = A.dtype
    n = len(b)
    A = A.copy()
    b = b.copy()
    for k in range(n - 1):
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        if np.abs(A[k, k]) < LIMIAR_SINGULARIDADE:
            raise ValueError(f"Matriz singular! Pivo A[{k},{k}] = {A[k,k]:.4e} < {LIMIAR_SINGULARIDADE:.0e}")

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i]     -= m * b[k]

    if np.abs(A[n-1, n-1]) < LIMIAR_SINGULARIDADE:
        raise ValueError(f"Matriz singular! Pivo A[{n-1},{n-1}] = {A[n-1,n-1]:.4e} < {LIMIAR_SINGULARIDADE:.0e}")
    return A, b

def resolver_gauss_singular(A, b):
    """ Resolve Ax = b com bloqueio contra matrizes singulares. """
    try:
        Au, bu = gauss_singular(A, b)
        return subst_retro(Au, bu)
    except ValueError as e:
        print(f"Erro ao resolver o sistema: {e}")
        return np.full_like(b, np.nan)
