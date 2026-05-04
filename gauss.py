"""
Módulo: gauss.py
Descrição: Implementa a Eliminação de Gauss clássica, a Eliminação de Gauss 
com pivoteamento parcial e uma versão com detecção de singularidade.
"""

import numpy as np

def subst_retro (A , b ):
    """ Substituição retroativa para sistema triangular superior ."""
    n = len (b )
    x = np . zeros (n )
    for i in range ( n - 1, -1, -1) :
        x[ i] = ( b[i ] - A[i , i +1:] @ x [i +1:]) / A[i , i]
    return x

def gauss_sem_pivoteamento (A , b) :
    """ Eliminação de Gauss sem pivoteamento.
    Retorna ( A_triangular , b_modificado ).
    """
    A = np . array (A , dtype = np.float32 )
    b = np . array (b , dtype = np.float32 )
    n = len (b )

    for k in range ( n - 1) :
        for i in range (k + 1 , n) :
            m = A [i , k ] / A[k , k ] # multiplicador
            A [i , k :] -= m * A [k , k :]
            b [i ] -= m * b [k ]

    return A , b

def resolver_gauss_sem_pivoteamento(A, b):
    """ Resolve Ax = b usando Gauss sem pivoteamento. """
    Au, bu = gauss_sem_pivoteamento(A, b)
    return subst_retro(Au, bu)

def gauss (A , b) :
    """ Eliminação de Gauss com pivoteamento parcial .
    Retorna ( A_triangular , b_modificado ).
    """
    A = np . array (A , dtype = np.float32 )
    b = np . array (b , dtype = np.float32 )
    n = len (b )

    for k in range ( n - 1) :
        # Pivoteamento parcial : busca maior | a_ik | abaixo da
        # linha k

        p = np.argmax ( np.abs ( A[ k: , k])) + k
        A [[k, p]] = A [[p, k]]
        b [[k, p]] = b [[p, k]]

        for i in range (k + 1 , n) :
            m = A [i , k ] / A[k , k ] # multiplicador
            A [i , k :] -= m * A [k , k :]
            b [i ] -= m * b [k ]

    return A , b

def resolver_gauss(A, b):
    """ Resolve Ax = b usando Gauss com pivoteamento parcial. """
    Au, bu = gauss(A, b)
    return subst_retro(Au, bu)

def gauss_sem_pivoteamento_precision(A, b, precision):
    """ Eliminação de Gauss sem pivoteamento com precisão customizável.
    Retorna (A_triangular, b_modificado).
    """
    A = np.array(A, dtype=precision)
    b = np.array(b, dtype=precision)
    n = len(b)

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]  # multiplicador
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return A, b

def resolver_gauss_sem_pivoteamento_precision(A, b, precision):
    """ Resolve Ax = b usando Gauss sem pivoteamento com precisão customizável. """
    Au, bu = gauss_sem_pivoteamento_precision(A, b, precision)
    return subst_retro(Au, bu)

def gauss_precision(A, b, precision):
    """ Eliminação de Gauss com pivoteamento parcial e precisão customizável.
    Retorna (A_triangular, b_modificado).
    """
    A = np.array(A, dtype=precision)
    b = np.array(b, dtype=precision)
    n = len(b)

    for k in range(n - 1):
        # Pivoteamento parcial: busca maior |a_ik| abaixo da linha k
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]  # multiplicador
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return A, b

def resolver_gauss_precision(A, b, precision):
    """ Resolve Ax = b usando Gauss com pivoteamento parcial e precisão customizável. """
    Au, bu = gauss_precision(A, b, precision)
    return subst_retro(Au, bu)

def experimento_estabilidade_gauss(a11_vals):
    """
    Executa o experimento de estabilidade (Q1.4) variando a11.
    """
    erro_sem, erro_com = [], []
    x_true = np.array([1/3, 2/3], dtype=np.float32)

    for a11 in a11_vals:
        # Matriz A variando a11
        A = np.array([[a11, 3], [1, 1]], dtype=np.float32)
        b = np.array([2.0001, 1], dtype=np.float32)
        
        xs = resolver_gauss_sem_pivoteamento(A, b)
        xc = resolver_gauss(A, b)
        
        # Erro relativo em x1
        erro_sem.append(abs(xs[0] - x_true[0]) / x_true[0])
        erro_com.append(abs(xc[0] - x_true[0]) / x_true[0])
        
    return erro_sem, erro_com

LIMIAR_SINGULARIDADE = 1e-12

def gauss_singular(A, b):
    """ Eliminação de Gauss com detecção rigorosa de quase-singularidade. """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    for k in range(n - 1):
        p = np.argmax(np.abs(A[k:, k])) + k
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        if np.abs(A[k, k]) < LIMIAR_SINGULARIDADE:
            raise ValueError(f"Matriz singular! Pivô A[{k},{k}] = {A[k,k]:.4e} < {LIMIAR_SINGULARIDADE:.0e}")

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    if np.abs(A[n-1, n-1]) < LIMIAR_SINGULARIDADE:
        raise ValueError(f"Matriz singular! Pivô A[{n-1},{n-1}] = {A[n-1,n-1]:.4e} < {LIMIAR_SINGULARIDADE:.0e}")
    
    return A, b

def resolver_gauss_singular(A, b):
    """ Resolve Ax = b com bloqueio contra matrizes singulares. """
    try:
        Au, bu = gauss_singular(A, b)
        return subst_retro(Au, bu)
    except ValueError as e:
        print(f"Erro ao resolver o sistema: {e}")
        return np.full_like(b, np.nan)
