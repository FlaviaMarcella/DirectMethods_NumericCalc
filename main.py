"""
Módulo: main.py
Plano de Investigação: Sistemas Lineares
Disciplina: Cálculo Numérico
Descrição: Script principal exigido na entrega. Reproduz todos os testes
do relatório e gera os gráficos necessários em 'resultados.pdf'.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

# Importando os módulos desenvolvidos
from gauss import resolver_gauss, gauss, resolver_gauss_sem_pivoteamento, gauss_float32, resolver_gauss_float32, resolver_gauss_sem_pivoteamento_float32, resolver_gauss_Singularidade
from lu import fatoracao_lu, fatoracao_lu_vectorized, subst_prog, subst_retro, resolver_lu
from cholesky import cholesky
from pagerank import calcular_pagerank

def main():
    # Garantir reprodutibilidade global
    np.random.seed(42)

    # Setup da figura para resultados.pdf (Grelha 4x2 para os 7 gráficos)
    fig, axs = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle('Plano de Investigação: Sistemas Lineares - Resultados Consolidados', fontsize=18, fontweight='bold')

    # =========================================================================
    # Seção 1: Eliminação de Gauss e Pivoteamento
    # =========================================================================
    print("--- Seção 1: Eliminação de Gauss e Pivoteamento ---")
    A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=np.float64)
    b = np.array([1, 2, 3], dtype=np.float64)
    x = resolver_gauss(A, b)
    residuo = np.linalg.norm(A @ x - b)

    print(f"    Seção 1 - Matriz A:\n{A}")
    print(f"    Seção 1 - Vetor b: {b}")

    print(f"    Seção 1 - Residuo: {residuo}")
    print(f"    Seção 1 - Solução x: {x}")

    # Verifique a solução (opcional)
    # A @ x deve ser aproximadamente igual a b
    print(f"    Seção 1 - Verificação (Ax): {A @ x}")
    print(f"    Seção 1 - Diferença (Ax - b): {A @ x - b}")

    Au, bu = gauss(A, b)
    print(f"    Seção 1 - Matriz Triangular Superior (U):\n{Au}")

    A = np.array([[0.0003, 3], [1, 1]], dtype=np.float32)
    b = np.array([2.0001, 1], dtype=np.float32)

    x_sem_pivoteamento = resolver_gauss_sem_pivoteamento(A, b)
    x_com_pivoteamento = resolver_gauss(A, b)

    # Resíduo -> Ax - b
    residuo_sem_pivoteamento = np.linalg.norm(A @ x_sem_pivoteamento - b)
    residuo_com_pivoteamento = np.linalg.norm(A @ x_com_pivoteamento - b)

    print(f"    Seção 1 - Matriz A:\n{A}")
    print(f"    Seção 1 - Vetor b: {b}")

    print(f"    Seção 1 - Residuo sem Pivot: {residuo_sem_pivoteamento}")
    print(f"    Seção 1 - Residuo com Pivot: {residuo_com_pivoteamento}")

    print(f"    Seção 1 - Solução x sem pivoteamento: {x_sem_pivoteamento}")
    print(f"    Seção 1 - Solução x com pivoteamento: {x_com_pivoteamento}")

    # Calcular os valores verdadeiros de x1 e x2 para comparação
    x1_true = 1/3
    x2_true = 2/3
    x_true = np.array([x1_true, x2_true])

    # Calcular o erro relativo sem pivoteamento
    erro_relativo_sem_pivoteamento_x1 = np.abs(x_sem_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
    erro_relativo_sem_pivoteamento_x2 = np.abs(x_sem_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])

    # Calcular o erro relativo com pivoteamento
    erro_relativo_com_pivoteamento_x1 = np.abs(x_com_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
    erro_relativo_com_pivoteamento_x2 = np.abs(x_com_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])

    print(f"    Seção 1 - Erro Relativo Sem Pivoteamento")
    print(f"    Seção 1 - Erro Relativo X1: {erro_relativo_sem_pivoteamento_x1}")
    print(f"    Seção 1 - Erro Relativo X2: {erro_relativo_sem_pivoteamento_x2}")

    print(f"    Seção 1 - Erro Relativo Com Pivoteamento")
    print(f"    Seção 1 - Erro Relativo X1: {erro_relativo_com_pivoteamento_x1}")
    print(f"    Seção 1 - Erro Relativo X2: {erro_relativo_com_pivoteamento_x2}")

    # Efeito do Pivoteamento para diferentes valores de a11 (Q1.4)
    a11_values = [0.1, 0.001, 0.000001, 0.000000001]

    # Listas para armazenar os resultados para plotagem
    erros_relativos_sem_pivoteamento_x1_list = []
    erros_relativos_com_pivoteamento_x1_list = []
    residuos_sem_pivot_list = []
    residuos_com_pivot_list = []

    # Calcular os valores verdadeiros de x1 e x2 para comparação
    x1_true = 1/3
    x2_true = 2/3
    x_true = np.array([x1_true, x2_true])

    print(f"    Seção 1 - Valores verdadeiros x_true: {x_true}")

    for a11_current in a11_values:
        print(f"\n     Seção 1 - Testando com a11 = {a11_current:.0e} ---")

        # Defina a matriz A e o vetor b para um sistema de equações lineares
        A = np.array(
            [
                [a11_current, 3],
                [1, 1]
            ], dtype=np.float32)

        b = np.array([2.0001, 1], dtype=np.float32)

        # Resolva o sistema usando a função resolver_gauss_sem_pivoteamento
        x_solucao_sem_pivoteamento = resolver_gauss_sem_pivoteamento_float32(A, b)
        x_solucao_com_pivoteamento = resolver_gauss_float32(A, b)

        # Resíduo -> ∥Ax − b∥2
        residuo_sem_pivot =  np.linalg.norm(A @ x_solucao_sem_pivoteamento - b)
        residuo_com_pivot =  np.linalg.norm(A @ x_solucao_com_pivoteamento - b)

        # Calcular o erro relativo sem pivoteamento
        erro_relativo_sem_pivoteamento_x1 = np.abs(x_solucao_sem_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
        erro_relativo_sem_pivoteamento_x2 = np.abs(x_solucao_sem_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])

        # Calcular o erro relativo com pivoteamento
        erro_relativo_com_pivoteamento_x1 = np.abs(x_solucao_com_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
        erro_relativo_com_pivoteamento_x2 = np.abs(x_solucao_com_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])

        # Armazenar resultados
        residuos_sem_pivot_list.append(residuo_sem_pivot)
        residuos_com_pivot_list.append(residuo_com_pivot)
        erros_relativos_sem_pivoteamento_x1_list.append(erro_relativo_sem_pivoteamento_x1)
        erros_relativos_com_pivoteamento_x1_list.append(erro_relativo_com_pivoteamento_x1)

        print(f"    Seção 1 - Matriz A:\n{A}")
        print(f"    Seção 1 - Vetor b: {b}")
        print(f"    Seção 1 - A11: {a11_current}")
        print(f"    Seção 1 - Residuo sem Pivot: {residuo_sem_pivot}")
        print(f"    Seção 1 - Residuo com Pivot: {residuo_com_pivot}")
        print(f"    Seção 1 - Solução x sem pivoteamento: {x_solucao_sem_pivoteamento}")
        print(f"    Seção 1 - Solução x com pivoteamento: {x_solucao_com_pivoteamento}")
        print(f"    Seção 1 - Erro Relativo Sem Pivoteamento")
        print(f"    Seção 1 - Erro Relativo X1: {erro_relativo_sem_pivoteamento_x1}")
        print(f"    Seção 1 - Erro Relativo X2: {erro_relativo_sem_pivoteamento_x2}")
        print(f"    Seção 1 - Erro Relativo Com Pivoteamento")
        print(f"    Seção 1 - Erro Relativo X1: {erro_relativo_com_pivoteamento_x1}")
        print(f"    Seção 1 - Erro Relativo X2: {erro_relativo_com_pivoteamento_x2}")

    # Plotar o erro relativo em x1 em função de a11 (escala log-log)
    # Nota: usar .close() ao invés de .show() para evitar bloqueios interativos
    axs[0, 0].loglog(a11_values, erros_relativos_sem_pivoteamento_x1_list, 'o-', label='Sem Pivoteamento', color='blue')
    axs[0, 0].loglog(a11_values, erros_relativos_com_pivoteamento_x1_list, 'x-', label='Com Pivoteamento', color='red')
    axs[0, 0].set_xlabel('Valor de a11 (Log Scale)')
    axs[0, 0].set_ylabel('Erro Relativo em x1 (Log Scale)')
    axs[0, 0].set_title('Q1.4: Efeito do Pivoteamento')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.7)

    print('    Seção 1 - Resolver_gauss (Sistema Singular Modificado) ---')

    # Defina a matriz A e o vetor b para um sistema de equações lineares
    A_singular = np.array([
        [1, -3, 1],
        [6, -18, 4],
        [-1, 3, -1]
    ], dtype=np.float64)

    b_singular = np.array([1, 2, 4], dtype=np.float64)

    # Resolva o sistema usando a função resolver_gauss
    x_solucao_singular = resolver_gauss_Singularidade(A_singular, b_singular)

    # resíduo -> ∥Ax − b∥2
    # Se x_solucao_singular contém NaN, o resíduo também será NaN
    residuo_singular = np.linalg.norm(A_singular @ x_solucao_singular - b_singular) if not np.isnan(x_solucao_singular).any() else np.nan

    print(f"    Seção 1 - Matriz A Singular:\n{A_singular}")
    print(f"    Seção 1 - Vetor b Singular: {b_singular}")

    print(f"    Seção 1 - Residuo: {residuo_singular}")
    print(f"    Seção 1 - Solução x: {x_solucao_singular}")

    # Verifique a solução (opcional)
    # A @ x deve ser aproximadamente igual a b
    print(f"    Seção 1 - Verificação (Ax): {A_singular @ x_solucao_singular}")
    print(f"    Seção 1 - Diferença (Ax - b): {A_singular @ x_solucao_singular - b_singular}")

    # =========================================================================
    # Seção 2: Fatoração LU
    # =========================================================================
    print("--- Seção 2: Fatoração LU vs Gauss Repetido (Q3.2) ---")

    A = np.array([
        [2, 1, 1], 
        [4, -6, 0], 
        [-2, 7 ,2]], dtype=np.float64)

    L, U = fatoracao_lu(A)

    erro_fatoracao = np.linalg.norm(L @ U - A, ord='fro')
    print(f"    Seção 2 - Matriz A:\n{A}")
    print(f"    Seção 2 - Matriz L:\n{L}")
    print(f"    Seção 2 - Matriz U:\n{U}")
    print(f"    Seção 2 - Erro na Fatoração LU: {erro_fatoracao}")

    b1 = np.array([1, 2, 3], dtype=np.float64)
    b2 = np.array([0, 1, -1], dtype=np.float64)

    # Resolvendo para B1
    y1 = subst_prog(L, b1)
    x1 = subst_retro(U, y1)
    residuo1 = np.linalg.norm(A @ x1 - b1)
    print(f"    Seção 2 - Vetor b1: {b1}")
    print(f"    Seção 2 - Solução x para b1: {x1}")
    print(f"    Seção 2 - Residuo para b1: {residuo1}")
    print(f"    Seção 2 - Verificação (Ax para b1): {A @ x1}")
    print(f"    Seção 2 - Diferença (Ax - b1): {A @ x1 - b1}")

    # Resolvendo para B2
    y2 = subst_prog(L, b2)
    x2 = subst_retro(U, y2)
    residuo2 = np.linalg.norm(A @ x2 - b2)
    print(f"    Seção 2 - Vetor b2: {b2}")
    print(f"    Seção 2 - Solução x para b2: {x2}")
    print(f"    Seção 2 - Residuo para b2: {residuo2}")
    print(f"    Seção 2 - Verificação (Ax para b2): {A @ x2}")
    print(f"    Seção 2 - Diferença (Ax - b2): {A @ x2 - b2}")

    n = 200
    k = 50
    np.random.seed(42)
    A = np.random.rand(n, n)
    A += np.eye(n) * n

    b = [np.random.rand(n) for _ in range(k)]

    print(f"    Seção 2 - Experimento de Desempenho: Fatoração LU vs Gauss para {k} vetores b")
    print(f"    Seção 2 - Dimensão da Matriz A: {A.shape}")
    print(f"    Seção 2 - Número de vetores b: {k}")

    # Gauss repetido
    print("    Seção 2 - Resolvendo com Gauss repetido...")
    start_time_gauss = time.time()

    x_solutions_gauss = []
    for i, b_vec in enumerate(b):
        x_solutions_gauss.append(resolver_gauss(A.copy(), b_vec.copy()))
    end_time_gauss = time.time()
    time_gauss = end_time_gauss - start_time_gauss 
    print(f"    Seção 2 - Tempo total para Gauss repetido: {time_gauss:.4f} segundos")

    # Fatoração LU
    print("    Seção 2 - Resolvendo com Fatoração LU...")
    start_time_lu = time.time()
    L, U = fatoracao_lu(A)
    x_solutions_lu = []
    for i, b_vec in enumerate(b):
        y = subst_prog(L, b_vec)
        x = subst_retro(U, y)
        x_solutions_lu.append(x)
    end_time_lu = time.time()
    time_lu = end_time_lu - start_time_lu
    print(f"    Seção 2 - Tempo total para Fatoração LU: {time_lu:.4f} segundos")

    print(f"    Seção 2 - Comparação de Desempenho")
    print(f"    Seção 2 - Tempo Gauss repetido: {time_gauss:.4f} segundos")
    print(f"    Seção 2 - Tempo Fatoração LU: {time_lu:.4f} segundos")
    print(f"    Seção 2 - Speedup (Gauss vs LU): {time_gauss / time_lu:.2f}x mais rápido com LU")

    A = np.array([
        [0, 1],
        [2, 3]
    ], dtype=np.float64)

    try:
        L, U = fatoracao_lu(A)
        erro_fatoracao = np.linalg.norm(L @ U - A, ord='fro')

        print(f"    Seção 2 - Fatoração LU sem pivoteamento")
        print(f"    Seção 2 - Matriz A Singular:\n{A}")
        print(f"    Seção 2 - Matriz L:\n{L}")
        print(f"    Seção 2 - Matriz U:\n{U}")
        print(f"    Seção 2 - Erro na Fatoração LU: {erro_fatoracao}")
    except ValueError as e:
        print(f"    Seção 2 - Erro durante a fatoração LU: {e}")
        erro_fatoracao = np.nan
        print(f"    Seção 2 - Nota: Fatoração LU com pivoteamento resolveria este problema.")
        print(f"    Seção 2 - A matriz tem um zero na primeira posição diagonal, exigindo troca de linhas.")

    # =========================================================================
    # Seção 3: Fatoração Cholesky (Gráfico Q3.2)
    # =========================================================================
    print("--- Seção 3: Fatoração Cholesky ---")

    A_1 = np.array([
        [4, 2],
        [2, 3]
    ], dtype=np.float64)

    A_2 = np.array([
        [1, 2],
        [2, 1]
    ], dtype=np.float64)

    A_3 = np.array([
        [4, 2 ,2],
        [2, 3, 0],
        [2, 0, 3]
    ], dtype=np.float64)

    print(f"    Seção 3 - Fatoração Cholesky para Matriz A_1:")
    print(f"    Seção 3 - Matriz A_1:\n{A_1}")
    print("     Seção 3 - Previsão: A_1 é simétrica (já que A=A.T) e deve ser positiva definida, pois seus menores principais são positivos. Cholesky deve ter sucesso.")
    
    try:
        L1 = cholesky(A_1)
        print("Resultado Cholesky (A1): Sucesso!")
        print(f"Matriz L1:\n{L1}")
    except ValueError as e:
        print(f"Resultado Cholesky (A1): Falha - {e}")
    eigvals1 = np.linalg.eigvalsh(A_1)
    print(f"Autovalores de A1: {eigvals1}")
    print(f"Critério SPD: Todos os autovalores são positivos ({np.all(eigvals1 > 0)}), confirmando que A1 é SPD.\n")

    print(f"    Seção 3 - Matriz A_2:\n{A_2}")
    print(f"    Seção 3 - Matriz A_3:\n{A_3}")

    print("     Seção 3 - Fatoração Cholesky para Matriz A_2:")
    print(f"     Seção 3 - Matriz A_2:\n{A_2}")
    print("     Seção 3 - Previsão: A_2 é simétrica, mas não é positiva definida (determinante é negativo, autovalores devem ter sinais diferentes). Cholesky deve falhar.")

    try:
        L2 = cholesky(A_2)
        print("Resultado Cholesky (A_2): Sucesso!")
        print(f"Matriz L2:\n{L2}")
    except ValueError as e:
        print(f"Resultado Cholesky (A_2): Falha - {e}")
    eigvals2 = np.linalg.eigvalsh(A_2)
    print(f"Autovalores de A_2: {eigvals2}")
    print(f"Critério SPD: Um autovalor é negativo ({np.all(eigvals2 > 0)}), confirmando que A_2 não é SPD.\n")

    print("     Seção 3 - Fatoração Cholesky para Matriz A_3:")
    print("     Seção 3 - Matriz A_3:\n{A_3}")
    print("     Seção 3 - Previsão: A_3 é simétrica e deve ser positiva definida. Seus menores principais devem ser positivos. Cholesky deve ter sucesso.")

    try:
        L3 = cholesky(A_3)
        print("Resultado Cholesky (A_3): Sucesso!")
        print(f"Matriz L3:\n{L3}")
    except ValueError as e:
        print(f"Resultado Cholesky (A_3): Falha - {e}")
    eigvals3 = np.linalg.eigvalsh(A_3)
    print(f"Autovalores de A_3: {eigvals3}")
    print(f"Critério SPD: Todos os autovalores são positivos ({np.all(eigvals3 > 0)}), confirmando que A_3 é SPD.\n")

    ns = [50, 100, 200, 500]
    tempos_cholesky = []
    tempos_lu = []

    print("    Seção 3 - Comparação de Desempenho: Cholesky vs LU para matrizes SPD")
    for n in ns:
        print(f"    Seção 3 - Testando para n={n}...")
        B = np.random.rand(n, n).astype(np.float32)
        A = B @ B.T + np.eye(n, dtype=np.float32) * n  # Garantir que A é SPD
        
        start_time = time.perf_counter()
        try:
            L_cholesky = cholesky(A.copy()) # Usar .copy() para não modificar A original
            end_time = time.perf_counter()
            tempos_cholesky.append(end_time - start_time)
            print(f"  Tempo Cholesky: {tempos_cholesky[-1]:.6f} s")
        except ValueError as e:
            print(f"  Cholesky falhou para n={n}: {e}")
            tempos_cholesky.append(np.nan)

        # --- Medir tempo da Fatoração LU (sem pivoteamento) ---
        start_time = time.perf_counter()
        try:
            L_lu, U_lu = fatoracao_lu(A.copy()) # Usar .copy()
            end_time = time.perf_counter()
            tempos_lu.append(end_time - start_time)
            print(f"  Seção 3 - Tempo LU: {tempos_lu[-1]:.6f} s")
        except ValueError as e:
            print(f"  Seção 3 - LU falhou para n={n}: {e}")
            tempos_lu.append(np.nan)

    # --- Plotagem dos resultados ---
    axs[0, 1].loglog(ns, tempos_cholesky, 'o-', label='Cholesky', color='blue')
    axs[0, 1].loglog(ns, tempos_lu, 'x-', label='LU (sem pivoteamento)', color='red')
    axs[0, 1].set_xlabel('Dimensão da Matriz (n)')
    axs[0, 1].set_ylabel('Tempo de Execução (s)')
    axs[0, 1].set_title('Q3.2: Cholesky vs LU (SPD)')
    axs[0, 1].legend()
    axs[0, 1].grid(True, which="both", ls="-", alpha=0.7)

    # --- Confirmação Empírica da Relação de Tempos ---
    print("\n--- Relação de tempos (LU / Cholesky) ---")
    for i, n_val in enumerate(ns):
        if not np.isnan(tempos_cholesky[i]) and not np.isnan(tempos_lu[i]) and tempos_cholesky[i] > 0:
            ratio = tempos_lu[i] / tempos_cholesky[i]
            print(f"    Seção 3 - Para n={n_val}: Tempo LU / Tempo Cholesky = {ratio:.2f}x")
        else:
            print(f"    Seção 3 - Para n={n_val}: Não foi possível calcular a relação (tempo inválido).")
    # =========================================================================
    # Seção 4: Thomas vs. Gauss (Gráfico Q4.2) - TODO (Colega)
    # =========================================================================
    print("--- Seção 4: Thomas vs Gauss (Q4.2) --- [TODO: Colega]")
    axs[1, 1].text(0.5, 0.5, "  Tópico 4: Thomas vs Gauss\n(Parte do Colega)", ha='center', va='center')
    axs[1, 1].set_title('   Q4.2: Thomas vs Gauss')

    # =========================================================================
    # Seção 5: Custo computacional (Gráfico Q5.2) - Cholesky vs LU
    # =========================================================================

    # =========================================================================
    # Seção 6: Condicionamento (Gráfico Q6.1) - TODO (Colega)
    # =========================================================================
    print("--- Seção 6: Condicionamento (Q6.1) --- [TODO: Colega]")
    axs[2, 0].text(0.5, 0.5, "Tópico 6: Condicionamento\n(Parte do Colega)", ha='center', va='center')
    axs[2, 0].set_title('Q6.1: Condicionamento')

    # =========================================================================
    # Seção 7: PageRank (Gráfico Q7.3)
    # =========================================================================
    print("--- Seção 7: PageRank (Q7.3) ---")
    
    # (a) Matriz de Adjacência G e Construção de P
    G = np.array([
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0]
    ])

    # Normalizando as colunas para obter a matriz de transição P
    # P_ij = G_ij / sum(G[:, j])
    P = G / G.sum(axis=0)

    # (b) Montar o sistema (I - alpha*P.T) * pi = (1-alpha)/n * e
    n = 4
    alpha = 0.85
    I = np.eye(n)
    e = np.ones(n)

    A_page = I - alpha * P.T
    b_page = ((1 - alpha) / n) * e

    # (c) Resolver com resolver_gauss e resolver_lu
    pi_gauss = resolver_gauss(A_page, b_page)
    pi_lu, _, _ = resolver_lu(A_page, b_page)

    # (d) Verificar norma L1
    norm_l1 = np.linalg.norm(pi_gauss, 1)

    # (e) Ordenar páginas
    ranking = np.argsort(pi_gauss)[::-1]

    print(f"    Vetor de Ranks (Gauss): {pi_gauss}")
    print(f"    Soma dos Ranks (Norma L1): {norm_l1:.4f}")
    print(f"    Ranking das páginas (mais importante para menos): {ranking}")

    np.random.seed(0)
    n_20 = 20
    p_prob = 0.3
    alpha = 0.85

    # Gerar Grafo Aleatório
    G_20 = (np.random.rand(n_20, n_20) < p_prob).astype(float)
    # Garantir que não existam colunas nulas (dangling nodes)
    for j in range(n_20):
        if G_20[:, j].sum() == 0: G_20[0, j] = 1

    P_20 = G_20 / G_20.sum(axis=0)
    A_20 = np.eye(n_20) - alpha * P_20.T
    b_20 = ((1 - alpha) / n_20) * np.ones(n_20)

    # Tempo Gauss
    t0 = time.time()
    pi_gauss_20 = resolver_gauss(A_20, b_20)
    t_gauss = time.time() - t0
    res_gauss = np.linalg.norm(A_20 @ pi_gauss_20 - b_20)

    # Tempo LU (nossa implementação)
    t0 = time.time()
    pi_lu_20, _, _ = resolver_lu(A_20, b_20)
    t_lu = time.time() - t0
    res_lu = np.linalg.norm(A_20 @ pi_lu_20 - b_20)

    print(f"    Gauss Manual: Tempo={t_gauss:.6f}s, Resíduo={res_gauss:.2e}")
    print(f"    LU Manual:    Tempo={t_lu:.6f}s, Resíduo={res_lu:.2e}")

    alphas = [0.5, 0.7, 0.85, 0.95, 0.99]
    conds = []

    for a in alphas:
        M = np.eye(n_20) - a * P_20.T
        conds.append(np.linalg.cond(M, p=2))

    axs[2, 1].semilogy(alphas, conds, 'ro-')
    axs[2, 1].set_xlabel('Fator de Amortecimento (alpha)')
    axs[2, 1].set_ylabel('Número de Condicionamento Kappa_2 (Log)')
    axs[2, 1].set_title('Q7.3: Condicionamento do PageRank')
    axs[2, 1].grid(True)
    # =========================================================================
    # Seção 8: Impacto da Vetorização (Gráfico Q8.1)
    # =========================================================================
    print("--- Seção 8: Impacto da Vetorização (Q8.1) ---")
    
    epsilons = [10**(-i) for i in range(1, 15)]
    errors = []
    cond_numbers = []
    residuos = []
    valid_eps = []

    x_true = np.array([1.0, 1.0])

    for eps in epsilons:
        try:
            # Criamos a matriz A_eps
            A_eps = np.array([[1.0, 1.0], [1.0, 1.0 + eps]], dtype=np.float64)
            b = A_eps @ x_true

            # Resolver usando a função de Gauss implementada anteriormente
            x_sol = resolver_gauss(A_eps, b)

            # Cálculos de métricas
            cond_numbers.append(np.linalg.cond(A_eps))
            errors.append(np.linalg.norm(x_sol - x_true))
            residuos.append(np.linalg.norm(A_eps @ x_sol - b))
            valid_eps.append(eps)
        except ValueError:
            print(f"Limite de precisão atingido para epsilon = {eps}")
            break

    # Visualização
    plt.figure(figsize=(10, 5))
    plt.loglog(valid_eps, errors, 'o-', label='Erro da Solução (||x - x_true||)')
    plt.loglog(valid_eps, cond_numbers, 's--', label='Número de Condicionamento κ(A)')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Magnitude')
    plt.title('Impacto da Quase-Singularidade na Estabilidade Numérica')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.show()

    def build_poisson_matrix(m):
        n = m * m
        # Construção da matriz laplaciana 2D (stencil de 5 pontos)
        main_diag = np.ones(n) * 4
        off_diag = np.ones(n - 1) * -1
        for i in range(1, m):
            off_diag[i * m - 1] = 0

        # Matriz densa
        A_dense = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        A_dense += np.diag(np.ones(n - m) * -1, m) + np.diag(np.ones(n - m) * -1, -m)

        # Matriz esparsa
        A_sparse = sp.csr_matrix(A_dense)
        return A_dense, A_sparse

    m = 10
    n = m**2
    A_dense, A_sparse = build_poisson_matrix(m)
    b = np.ones(n)

    # Resolver com Gauss Manual
    t0 = time.time()
    x_gauss_p = resolver_gauss(A_dense, b)
    t_gauss_p = time.time() - t0

    # Resolver com Scipy Sparse
    t0 = time.time()
    x_sparse_p = spsolve(A_sparse, b)
    t_sparse_p = time.time() - t0

    print(f"Dimensão do sistema Poisson: {n}x{n}")
    print(f"Tempo Gauss Manual (Denso): {t_gauss_p:.6f}s")
    print(f"Tempo Scipy spsolve (Esparso): {t_sparse_p:.6f}s")
    print(f"Aceleração: {t_gauss_p / t_sparse_p:.2f}x")

    # Benchmarking com n = 100
    n_bench = 100
    A_bench = np.random.rand(n_bench, n_bench).astype(np.float32)

    t0 = time.time()
    L_old, U_old = fatoracao_lu(A_bench)
    t_old = time.time() - t0

    t0 = time.time()
    L_new, U_new = fatoracao_lu_vectorized(A_bench)
    t_new = time.time() - t0

    print(f"Tamanho da Matriz: {n_bench}x{n_bench}")
    print(f"Tempo LU (Loops Python): {t_old:.6f}s")
    print(f"Tempo LU (Vetorizado): {t_new:.6f}s")
    print(f"Aceleração: {t_old / t_new:.2f}x")



    # Eixo 3,1 fica vazio ou para anotações
    axs[3, 1].axis('off')
    axs[3, 1].text(0.1, 0.5, "Relatório Completo\nCálculo Numérico 2026\nUNIFAL-MG\n\nAlunos:\nFlávia Marcella\nRenan Catini", 
                   fontsize=12, style='italic', bbox={'facecolor': 'grey', 'alpha': 0.1, 'pad': 10})

    # Salvando resultados
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("resultados.pdf", dpi=150)
    print("\n[!] INVESTIGAÇÃO CONCLUÍDA. Verifique o arquivo 'resultados.pdf'.")

if __name__ == "__main__":
    main()
