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
from formatacao import (secao_titulo, subsecao, destaque_resultado, destaque_erro, 
                       destaque_aviso, info_matriz, comparacao_metodos, relatorio_final, Cores)

def main():
    # Garantir reprodutibilidade global
    np.random.seed(42)

    # Setup da figura para resultados.pdf (Grelha 4x2 para os 7 gráficos)
    fig, axs = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle('Plano de Investigação: Sistemas Lineares - Resultados Consolidados', fontsize=18, fontweight='bold')

    # =========================================================================
    # Seção 1: Eliminação de Gauss e Pivoteamento
    # =========================================================================
    secao_titulo(1, "Eliminação de Gauss e Pivoteamento")
    
    subsecao("Teste 1: Sistema Simples 3x3")
    A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=np.float64)
    b = np.array([1, 2, 3], dtype=np.float64)
    x = resolver_gauss(A, b)
    residuo = np.linalg.norm(A @ x - b)
    
    info_matriz("Matriz A", A)
    print(f"    Vetor b: {b}")
    destaque_resultado("Resíduo", f"{residuo:.2e}")
    destaque_resultado("Solução x", str(x))
    
    # Verificação
    Ax = A @ x
    diferenca = Ax - b
    print(f"    Verificação (Ax): {Ax}")
    print(f"    Diferença (Ax - b): {diferenca}")
    
    # Matriz triangular superior
    Au, bu = gauss(A, b)
    info_matriz("Matriz Triangular Superior (U)", Au)

    subsecao("Teste 2: Efeito do Pivoteamento (Matriz Mal-Condicionada)")
    A = np.array([[0.0003, 3], [1, 1]], dtype=np.float32)
    b = np.array([2.0001, 1], dtype=np.float32)
    
    info_matriz("Matriz A", A)
    print(f"    Vetor b: {b}")

    x_sem_pivoteamento = resolver_gauss_sem_pivoteamento(A, b)
    x_com_pivoteamento = resolver_gauss(A, b)

    residuo_sem_pivoteamento = np.linalg.norm(A @ x_sem_pivoteamento - b)
    residuo_com_pivoteamento = np.linalg.norm(A @ x_com_pivoteamento - b)

    print(f"\n  {Cores.VERMELHO}[SEM PIVOTEAMENTO]:{Cores.RESET}")
    destaque_resultado("  Resíduo", f"{residuo_sem_pivoteamento:.6e}")
    destaque_resultado("  Solução x", str(x_sem_pivoteamento))
    
    print(f"\n  {Cores.VERDE}[COM PIVOTEAMENTO]:{Cores.RESET}")
    destaque_resultado("  Resíduo", f"{residuo_com_pivoteamento:.6e}")
    destaque_resultado("  Solução x", str(x_com_pivoteamento))
    
    # Cálculo de erros relativos
    x1_true = 1/3
    x2_true = 2/3
    x_true = np.array([x1_true, x2_true])
    
    erro_relativo_sem_pivoteamento_x1 = np.abs(x_sem_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
    erro_relativo_sem_pivoteamento_x2 = np.abs(x_sem_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])
    
    erro_relativo_com_pivoteamento_x1 = np.abs(x_com_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
    erro_relativo_com_pivoteamento_x2 = np.abs(x_com_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])
    
    print(f"\n  {Cores.AMARELO}[ERROS RELATIVOS]:{Cores.RESET}")
    print(f"    Sem Pivoteamento: x₁={erro_relativo_sem_pivoteamento_x1:.6e}, x₂={erro_relativo_sem_pivoteamento_x2:.6e}")
    print(f"    Com Pivoteamento: x₁={erro_relativo_com_pivoteamento_x1:.6e}, x₂={erro_relativo_com_pivoteamento_x2:.6e}")

    print(f"\n    {Cores.CIANO}Valores verdadeiros: x₁={x1_true:.6f}, x₂={x2_true:.6f}{Cores.RESET}")

    # Calcular o erro relativo com pivoteamento
    erro_relativo_com_pivoteamento_x1 = np.abs(x_com_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
    erro_relativo_com_pivoteamento_x2 = np.abs(x_com_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])

    # Efeito do Pivoteamento para diferentes valores de a11 (Q1.4)
    a11_values = [0.1, 0.001, 0.000001, 0.000000001]

    # Listas para armazenar os resultados para plotagem
    erros_relativos_sem_pivoteamento_x1_list = []
    erros_relativos_com_pivoteamento_x1_list = []
    residuos_sem_pivot_list = []
    residuos_com_pivot_list = []

    # Valores verdadeiros
    x1_true = 1/3
    x2_true = 2/3
    x_true = np.array([x1_true, x2_true])

    for a11_current in a11_values:
        print(f"\n  {Cores.BOLD}─ Testando com a₁₁ = {a11_current:.0e}{Cores.RESET}")
        
        # Defina a matriz A e o vetor b para um sistema de equações lineares
        A = np.array(
            [
                [a11_current, 3],
                [1, 1]
            ], dtype=np.float32)

        b = np.array([2.0001, 1], dtype=np.float32)
        
        info_matriz("    Matriz A", A)
        print(f"    Vetor b: {b}")

        # Resolver o sistema usando a função resolver_gauss_sem_pivoteamento
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

        print(f"    {Cores.VERMELHO}Sem Pivoteamento:{Cores.RESET} Resíduo={residuo_sem_pivot:.6e}, Erros=[{erro_relativo_sem_pivoteamento_x1:.6e}, {erro_relativo_sem_pivoteamento_x2:.6e}]")
        print(f"    {Cores.VERDE}Com Pivoteamento:{Cores.RESET}  Resíduo={residuo_com_pivot:.6e}, Erros=[{erro_relativo_com_pivoteamento_x1:.6e}, {erro_relativo_com_pivoteamento_x2:.6e}]")

        # Armazenar resultados
        residuos_sem_pivot_list.append(residuo_sem_pivot)
        residuos_com_pivot_list.append(residuo_com_pivot)
        erros_relativos_sem_pivoteamento_x1_list.append(erro_relativo_sem_pivoteamento_x1)
        erros_relativos_com_pivoteamento_x1_list.append(erro_relativo_com_pivoteamento_x1)

    # Plotar o erro relativo em x1 em função de a11 (escala log-log)
    # Nota: usar .close() ao invés de .show() para evitar bloqueios interativos
    axs[0, 0].loglog(a11_values, erros_relativos_sem_pivoteamento_x1_list, 'o-', label='Sem Pivoteamento', color='blue')
    axs[0, 0].loglog(a11_values, erros_relativos_com_pivoteamento_x1_list, 'x-', label='Com Pivoteamento', color='red')
    axs[0, 0].set_xlabel('Valor de a11 (Log Scale)')
    axs[0, 0].set_ylabel('Erro Relativo em x1 (Log Scale)')
    axs[0, 0].set_title('Q1.4: Efeito do Pivoteamento')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.7)

    # Teste com sistema singular
    subsecao("Teste 3: Sistema Singular")
    A_singular = np.array([
        [1, -3, 1],
        [6, -18, 4],
        [-1, 3, -1]
    ], dtype=np.float64)

    b_singular = np.array([1, 2, 4], dtype=np.float64)
    
    info_matriz("Matriz A (Singular)", A_singular)
    print(f"    Vetor b: {b_singular}")

    # Resolver (resultado esperado: NaN devido à singularidade)
    try:
        x_solucao_singular = resolver_gauss_Singularidade(A_singular, b_singular)
        destaque_resultado("Solução x", str(x_solucao_singular))
    except Exception as e:
        destaque_erro("Erro ao resolver", str(e))

    # =========================================================================
    # Seção 2: Fatoração LU
    # =========================================================================
    secao_titulo(2, "Fatoração LU")
    
    subsecao("Teste 1: Fatoração LU Básica com Duas RHS")
    A = np.array([
        [2, 1, 1], 
        [4, -6, 0], 
        [-2, 7 ,2]], dtype=np.float64)

    info_matriz("Matriz A", A)
    
    L, U = fatoracao_lu(A)
    
    info_matriz("Matriz L", L)
    info_matriz("Matriz U", U)
    
    erro_fatoracao = np.linalg.norm(L @ U - A, ord='fro')
    destaque_resultado("Erro na Fatoração", f"{erro_fatoracao:.2e}")

    # Resolver para dois vetores b diferentes
    b1 = np.array([1, 2, 3], dtype=np.float64)
    b2 = np.array([0, 1, -1], dtype=np.float64)
    
    # Solução com b1
    y1 = subst_prog(L, b1)
    x1 = subst_retro(U, y1)
    residuo_b1 = np.linalg.norm(A @ x1 - b1)
    
    print(f"    Vetor b₁: {b1}")
    destaque_resultado("Solução x₁", str(x1))
    destaque_resultado("Resíduo para b₁", f"{residuo_b1:.2e}")
    print(f"    Verificação (Ax₁): {A @ x1}")
    print(f"    Diferença (Ax₁ - b₁): {A @ x1 - b1}")
    
    # Solução com b2
    y2 = subst_prog(L, b2)
    x2 = subst_retro(U, y2)
    residuo_b2 = np.linalg.norm(A @ x2 - b2)
    
    print(f"\n    Vetor b₂: {b2}")
    destaque_resultado("Solução x₂", str(x2))
    destaque_resultado("Resíduo para b₂", f"{residuo_b2:.2e}")
    print(f"    Verificação (Ax₂): {A @ x2}")
    print(f"    Diferença (Ax₂ - b₂): {A @ x2 - b2}")

    subsecao("Teste 2: Desempenho - Gauss Repetido vs Fatoração LU")
    n = 200
    k = 50
    np.random.seed(42)
    A = np.random.rand(n, n)
    A += np.eye(n) * n

    b = [np.random.rand(n) for _ in range(k)]

    print(f"    Dimensão: {n}x{n} | Vetores b: {k}\n")

    # Gauss repetido
    start_time_gauss = time.time()
    x_solutions_gauss = [resolver_gauss(A.copy(), b_vec.copy()) for i, b_vec in enumerate(b)]
    time_gauss = time.time() - start_time_gauss

    # Fatoração LU
    start_time_lu = time.time()
    L, U = fatoracao_lu(A)
    x_solutions_lu = []
    for i, b_vec in enumerate(b):
        y = subst_prog(L, b_vec)
        x = subst_retro(U, y)
        x_solutions_lu.append(x)
    time_lu = time.time() - start_time_lu

    comparacao_metodos({
        'Gauss Repetido': {'tempo': time_gauss},
        'Fatoração LU': {'tempo': time_lu}
    })
    print(f"    {Cores.VERDE}[SPEEDUP] {time_gauss / time_lu:.2f}x mais rápido com LU{Cores.RESET}")

    subsecao("Teste 3: Fatoração com Matriz Singular")
    A_singular = np.array([
        [0, 1],
        [2, 3]
    ], dtype=np.float64)
    
    info_matriz("Matriz A (Singular)", A_singular)
    
    try:
        L, U = fatoracao_lu(A_singular)
        info_matriz("Matriz L", L)
        info_matriz("Matriz U", U)
    except ValueError as e:
        destaque_erro("Erro na Fatoração LU", str(e))

    # =========================================================================
    # Seção 3: Fatoração Cholesky (Gráfico Q3.2)
    # =========================================================================
    secao_titulo(3, "Fatoração Cholesky")
    
    subsecao("Teste 1: Matrizes SPD vs Não-SPD")

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

    # Teste A_1
    print(f"\n  {Cores.BOLD}Testando A_1:{Cores.RESET}")
    info_matriz("    Matriz A_1", A_1)
    eigenvalues_A1 = np.linalg.eigvalsh(A_1)
    print(f"        Autovalores: {eigenvalues_A1}")
    print(f"        É SPD? {np.all(eigenvalues_A1 > 0)}")
    
    try:
        L1 = cholesky(A_1)
        info_matriz("    Matriz L_1 (Cholesky)", L1)
        destaque_resultado("  Resultado", "Sucesso - A_1 é SPD")
    except ValueError as e:
        destaque_erro("  Resultado", str(e))

    # Teste A_2
    print(f"\n  {Cores.BOLD}Testando A_2:{Cores.RESET}")
    info_matriz("    Matriz A_2", A_2)
    eigenvalues_A2 = np.linalg.eigvalsh(A_2)
    print(f"        Autovalores: {eigenvalues_A2}")
    print(f"        É SPD? {np.all(eigenvalues_A2 > 0)}")
    
    try:
        L2 = cholesky(A_2)
        info_matriz("    Matriz L_2 (Cholesky)", L2)
        destaque_resultado("  Resultado", "Sucesso (não esperado)")
    except ValueError as e:
        print(f"        {Cores.VERDE}Falha esperada:{Cores.RESET} {e}")
        destaque_resultado("  Resultado", "Falhou (esperado)")

    # Teste A_3
    print(f"\n  {Cores.BOLD}Testando A_3:{Cores.RESET}")
    info_matriz("    Matriz A_3", A_3)
    eigenvalues_A3 = np.linalg.eigvalsh(A_3)
    print(f"        Autovalores: {eigenvalues_A3}")
    print(f"        É SPD? {np.all(eigenvalues_A3 > 0)}")
    
    try:
        L3 = cholesky(A_3)
        info_matriz("    Matriz L_3 (Cholesky)", L3)
        destaque_resultado("  Resultado", "Sucesso - A_3 é SPD")
    except ValueError as e:
        destaque_erro("  Resultado", str(e))

    subsecao("Teste 2: Desempenho - Cholesky vs LU para Matrizes SPD - Gráfico 3.2")
    ns = [50, 100, 200, 500]
    tempos_cholesky = []
    tempos_lu = []

    for n in ns:
        B = np.random.rand(n, n).astype(np.float32)
        A = B @ B.T + np.eye(n, dtype=np.float32) * n
        
        start_time = time.perf_counter()
        try:
            L_cholesky = cholesky(A.copy())
            tempos_cholesky.append(time.perf_counter() - start_time)
        except ValueError:
            tempos_cholesky.append(np.nan)

        start_time = time.perf_counter()
        try:
            L_lu, U_lu = fatoracao_lu(A.copy())
            tempos_lu.append(time.perf_counter() - start_time)
        except ValueError:
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
    subsecao("Relação de Tempos (LU / Cholesky)")
    for i, n_val in enumerate(ns):
        if not np.isnan(tempos_cholesky[i]) and not np.isnan(tempos_lu[i]) and tempos_cholesky[i] > 0:
            ratio = tempos_lu[i] / tempos_cholesky[i]
            status = f"{Cores.VERDE}~{ratio:.2f}x{Cores.RESET}" if ratio > 0.5 else f"{Cores.AMARELO}~{ratio:.2f}x{Cores.RESET}"
            print(f"    n={n_val:3d}: LU/Cholesky = {status}")
        else:
            print(f"    n={n_val:3d}: {Cores.VERMELHO}[Erro na medição]{Cores.RESET}")
    # =========================================================================
    # Seção 4: Thomas vs. Gauss (Gráfico Q4.2) - TODO (Colega)
    # =========================================================================
    secao_titulo(4, "Thomas vs Gauss")
    destaque_aviso("Parte do colega - Pendente")
    axs[1, 1].text(0.5, 0.5, "  Tópico 4: Thomas vs Gauss\n(Parte do Colega)", ha='center', va='center')
    axs[1, 1].set_title('   Q4.2: Thomas vs Gauss')

    # =========================================================================
    # Seção 5: Custo computacional (Gráfico Q5.2) - Cholesky vs LU
    # =========================================================================

    # =========================================================================
    # Seção 6: Condicionamento (Gráfico Q6.1) - TODO (Colega)
    # =========================================================================
    secao_titulo(6, "Condicionamento")
    destaque_aviso("Parte do colega - Pendente")
    axs[2, 0].text(0.5, 0.5, "Tópico 6: Condicionamento\n(Parte do Colega)", ha='center', va='center')
    axs[2, 0].set_title('Q6.1: Condicionamento')

    secao_titulo(7, "PageRank")
    
    subsecao("Teste 1: Sistema de 4 Páginas")
    
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
    ranking = np.argsort(pi_gauss)[::-1]

    print(f"    Vetor de Ranks (Gauss): {pi_gauss}")
    print(f"    Soma dos Ranks (Norma L1): {Cores.VERDE}{norm_l1:.4f}{Cores.RESET}")
    print(f"    Ranking (mais importante para menos): {ranking}")

    subsecao("Teste 2: Sistema de 20 Páginas")
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

    print(f"\n  Comparação (20 páginas): Gauss={t_gauss:.6f}s | LU={t_lu:.6f}s")

    # Condicionamento do PageRank
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
    secao_titulo(8, "Impacto da Vetorização")
    subsecao("Teste 1: Quase-Singularidade - Epsilon vs Erro")
    
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
            break

    print(f"    Epsilons testados: {len(valid_eps)} sucessos\n")
    axs[3, 0].loglog(valid_eps, errors, 'o-', label='Erro da Solução (||x - x_true||)')
    axs[3, 0].loglog(valid_eps, cond_numbers, 's--', label='Número de Condicionamento κ(A)')
    axs[3, 0].set_xlabel('Epsilon (ε)')
    axs[3, 0].set_ylabel('Magnitude')
    axs[3, 0].set_title('Q8.1 - Impacto da Quase-Singularidade na Estabilidade Numérica')
    axs[3, 0].legend()
    axs[3, 0].grid(True)
    axs[3, 0].invert_xaxis()

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

    subsecao("Teste 2: Poisson 2D (Denso vs Esparso)")
    print(f"    Dimensão: {n}x{n}\n")
    comparacao_metodos({
        'Gauss (Denso)': {'tempo': t_gauss_p},
        'Scipy spsolve (Esparso)': {'tempo': t_sparse_p}
    })
    print(f"    {Cores.VERDE}[ACELERACAO] {t_gauss_p / t_sparse_p:.2f}x com esparso{Cores.RESET}")

    # Benchmarking com n = 100
    subsecao("Teste 3: LU Vetorizado vs Loops Python")
    n_bench = 100
    A_bench = np.random.rand(n_bench, n_bench).astype(np.float32)

    t0 = time.time()
    L_old, U_old = fatoracao_lu(A_bench)
    t_old = time.time() - t0

    t0 = time.time()
    L_new, U_new = fatoracao_lu_vectorized(A_bench)
    t_new = time.time() - t0

    print(f"    Dimensão: {n_bench}x{n_bench}\n")
    comparacao_metodos({
        'LU (Loops Python)': {'tempo': t_old},
        'LU (Vetorizado)': {'tempo': t_new}
    })
    print(f"    {Cores.VERDE}[ACELERACAO] {t_old / t_new:.2f}x com vetorização{Cores.RESET}")



    # Eixo 3,1 fica vazio ou para anotações
    axs[3, 1].axis('off')
    axs[3, 1].text(0.1, 0.5, "Relatório Completo\nCálculo Numérico 2026\nUNIFAL-MG\n\nAlunos:\nFlávia Marcella\nRenan Catini", 
                   fontsize=12, style='italic', bbox={'facecolor': 'grey', 'alpha': 0.1, 'pad': 10})

    # Salvando resultados
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("resultados.pdf", dpi=150)
    relatorio_final(concluido=True)

if __name__ == "__main__":
    main()
