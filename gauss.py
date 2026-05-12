"""
Módulo: gauss.py
Descrição: Implementa a Eliminação de Gauss clássica, a Eliminação de Gauss 
com pivoteamento parcial e uma versão com detecção de singularidade.
Otimizado com vetorização NumPy para melhor desempenho.
"""

import numpy as np

def gauss (A , b) :
    """ Eliminação de Gauss com pivoteamento parcial .
    Retorna ( A_triangular , b_modificado ).
    """
    A = np . array (A)
    b = np . array (b)
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

def subst_retro (A , b ):
    """ Substitui o retroativa para sistema triangular
    # superior ."""
    n = len (b )
    x = np . zeros (n )
    for i in range ( n - 1, -1, -1) :
        x[ i] = ( b[i ] - A[i , i +1:] @ x [i +1:]) / A[i , i]
    return x

def resolver_gauss (A , b ):
    """ Pipeline completo : Gauss + substitui o retroativa ."""
    Au , bu = gauss (A , b)
    return subst_retro (Au , bu )

def gauss_sem_pivoteamento (A , b) :
    """ Elimina o de Gauss sem pivoteamento.
    Retorna ( A_triangular , b_modificado ).
    """
    A = np . array (A)
    b = np . array (b)
    n = len (b )

    for k in range ( n - 1) :
        for i in range (k + 1 , n) :
            m = A [i , k ] / A[k , k ] # multiplicador
            A [i , k :] -= m * A [k , k :]
            b [i ] -= m * b [k ]

    return A , b

def resolver_gauss_sem_pivoteamento (A , b ):
    """ Pipeline completo : Gauss + substitui o retroativa ."""
    Au , bu = gauss_sem_pivoteamento (A , b)
    return subst_retro (Au , bu )

def gauss_float32 (A , b) :
    """ Eliminação de Gauss com pivoteamento parcial .
    Retorna ( A_triangular , b_modificado ).
    """
    A = np . array (A, np.float32 )
    b = np . array (b, np.float32 )
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

def resolver_gauss_float32 (A , b ):
    """ Pipeline completo : Gauss + substitui o retroativa ."""
    Au , bu = gauss_float32 (A , b)
    return subst_retro (Au , bu )

def gauss_sem_pivoteamento_float32 (A , b) :
    """ Elimina o de Gauss sem pivoteamento.
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

def resolver_gauss_sem_pivoteamento_float32 (A , b ):
    """ Pipeline completo : Gauss + substitui o retroativa ."""
    Au , bu = gauss_sem_pivoteamento_float32 (A , b)
    return subst_retro (Au , bu )

def gauss_singular (A , b) :
    """ Eliminação de Gauss com pivoteamento parcial .
    Retorna ( A_triangular , b_modificado ).
    """
    A = np . array (A , dtype = float )
    b = np . array (b , dtype = float )
    n = len (b )

    for k in range ( n - 1) :
        # Pivoteamento parcial : busca maior | a_ik | abaixo da
        # linha k

        p = np.argmax ( np.abs ( A[ k: , k])) + k
        A [[k, p]] = A [[p, k]]
        b [[k, p]] = b [[p, k]]

        # *** Adição para detecção de singularidade ***
        # Se o pivô for muito pequeno, a matriz é singular ou quase singular.
        if np.abs(A[k, k]) < 1e-12: # Limiar para pivô muito pequeno
            raise ValueError(f"Matriz singular ou quase singular detectada! Pivô A[{k},{k}] = {A[k, k]} é muito pequeno.")

        for i in range (k + 1 , n) :
            m = A [i , k ] / A[k , k ] # multiplicador
            A [i , k :] -= m * A [k , k :]
            b [i ] -= m * b [k ]

    # *** Adição para checar o último pivô ***
    if np.abs(A[n-1, n-1]) < 1e-12:
        raise ValueError(f"Matriz singular ou quase singular detectada! Pivô A[{n-1},{n-1}] = {A[n-1, n-1]} é muito pequeno.")

    return A , b


def subst_retro_singular (A , b ):
    """ Substituição retroativa para sistema triangular
    # superior ."""
    n = len (b )
    x = np . zeros (n )
    for i in range ( n - 1, -1, -1) :
        # Tratar divisão por zero no caso de matrizes singulares
        if np.abs(A[i, i]) < 1e-12:
            raise ValueError(f"Divisão por zero ou por número muito pequeno na substituição retroativa! Pivô A[{i},{i}] = {A[i, i]}")
        x[ i] = ( b[i ] - A[i , i +1:] @ x [i +1:]) / A[i , i]
    return x


def resolver_gauss_Singularidade (A , b ):
    """ Pipeline completo : Gauss + substituição retroativa ."""
    try:
        Au , bu = gauss_singular (A , b)
        return subst_retro_singular (Au , bu )
    except ValueError as e:
        print(f"Erro ao resolver o sistema: {e}")
        return np.full_like(b, np.nan) # Retorna NaN se houver erro