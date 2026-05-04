"""
Módulo: lu.py
Descrição: Implementa o método de Fatoração LU de Doolittle (sem pivoteamento)
e as substituições progressiva e retroativa conforme notebook.
"""

import numpy as np
import time

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


def resolver_lu(A, b):
    """ Resolve Ax = b via fatoração LU.
    """
    L, U = fatoracao_lu(A)
    y = subst_prog(L, b)
    x = subst_retro(U, y)
    return x, L, U


def experimento_desempenho_lu(ns):
    """
    Compara o tempo de execução entre Gauss repetido e LU reutilizado.
    """
    from gauss import resolver_gauss
    
    np.random.seed(42)
    A_q2_3 = np.random.rand(n_c, n_c)
    
    # Adicionar uma pequena diagonal dominante para ajudar na estabilidade se necessário
    A_q2_3 += np.eye(n_c) * n_c # Isso ajuda a garantir que A não seja singular
    
    # Criar k vetores b aleatórios
    b_vectors = [np.random.rand(n_c) for _ in range(n_b)]

    # --- Estratégia (a): Gauss repetido ---
    print("\n--- Estratégia (a): Gauss repetido ---")
    t_start_gauss = time.time()

    x_solutions_gauss = []
    for i, b_vec in enumerate(b_vectors):
        x_sol = resolver_gauss(A_q2_3.copy(), b_vec.copy())
        x_solutions_gauss.append(x_sol)

    t_end_gauss = time.time()
    time_gauss = t_end_gauss - t_start_gauss
    print(f"Tempo de execução (Gauss repetido): {time_gauss:.6f} segundos")

    # --- Estratégia (b): LU reutilizado ---
    print("\n--- Estratégia (b): LU reutilizado ---")
    t_start_lu = time.time()

    # Fatorar A uma única vez
    L_q2_3, U_q2_3 = fatoracao_lu(A_q2_3.copy())

    x_solutions_lu = []
    for i, b_vec in enumerate(b_vectors):
        y_sol = subst_prog(L_q2_3, b_vec.copy())
        x_sol = subst_retro(U_q2_3, y_sol)
        x_solutions_lu.append(x_sol)

    t_end_lu = time.time()
    time_lu = t_end_lu - t_start_lu
    print(f"Tempo de execução (LU reutilizado): {time_lu:.6f} segundos")

    print("\n--- Comparação --- ")
    print(f"Gauss repetido: {time_gauss:.6f} s")
    print(f"LU reutilizado: {time_lu:.6f} s")
    print(f"LU reutilizado é aproximadamente {time_gauss / time_lu:.2f} vezes mais rápido que Gauss repetido.")
    
    return time_gauss, time_lu


def fatoracao_lu_vectorized(A):
    """ Fatoração LU (sem mudanças de vetorização nesta versão) """
    return fatoracao_lu(A)
