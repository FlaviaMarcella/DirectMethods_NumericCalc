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

# Importando os módulos desenvolvidos
from gauss import resolver_gauss, resolver_gauss_sem
from lu import fatoracao_lu, subst_prog, subst_retro, fatoracao_lu_vectorized
from cholesky import resolver_cholesky
from pagerank import calcular_pagerank

def main():
    # Configuração de subplots para o PDF final
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Resultados da Investigação - Sistemas Lineares', fontsize=16)

    # =================================================================
    # Secção 1: Verificação básica
    # =================================================================
    print("--- Secção 1: Verificação Básica ---")
    A1 = np.array([[3, 2, 4], 
                   [1, 1, 2], 
                   [4, 3, -2]], dtype=float)
    b1 = np.array([1, 2, 3], dtype=float)

    x1 = resolver_gauss(A1, b1)
    print(f"Secção 1 Solução: {x1}")
    print(f"Secção 1 Resíduo: {np.linalg.norm(A1 @ x1 - b1):.2e}\n")

    # =================================================================
    # Secção 2: Efeito do pivoteamento (Q1.3 e Q1.4)
    # =================================================================
    print("--- Secção 2: Efeito do Pivoteamento (float32) ---")
    
    A_mal = np.array([[0.0003, 3], 
                      [1, 1]], dtype=np.float32)
    b_mal = np.array([2.0001, 1], dtype=np.float32)
    
    x_sem = resolver_gauss_sem(A_mal.copy(), b_mal.copy())
    x_com = resolver_gauss(A_mal.copy(), b_mal.copy())
    
    print(f"Solução SEM Pivoteamento: {x_sem}")
    print(f"Solução COM Pivoteamento: {x_com}\n")

    # Gráfico (Q1.4) - Variação do a11 (Cálculo Dinâmico em float32)
    a11_vals = [1e-1, 1e-3, 1e-6, 1e-9]
    erro_sem_piv = []
    erro_com_piv = []
    x_true = np.array([1/3, 2/3], dtype=np.float32)

    for a11 in a11_vals:
        A_temp = np.array([[a11, 3], [1, 1]], dtype=np.float32)
        b_temp = np.array([2.0001, 1], dtype=np.float32)
        
        x_sem_temp = resolver_gauss_sem(A_temp, b_temp)
        x_com_temp = resolver_gauss(A_temp, b_temp)
        
        e_sem = max(abs(x_sem_temp[0] - x_true[0]) / abs(x_true[0]),
                    abs(x_sem_temp[1] - x_true[1]) / abs(x_true[1]))
        e_com = max(abs(x_com_temp[0] - x_true[0]) / abs(x_true[0]),
                    abs(x_com_temp[1] - x_true[1]) / abs(x_true[1]))
        
        erro_sem_piv.append(e_sem)
        erro_com_piv.append(e_com)

    axs[0, 0].plot(a11_vals, erro_sem_piv, marker='o', label='Sem Pivoteamento')
    axs[0, 0].plot(a11_vals, erro_com_piv, marker='s', linestyle='--', label='Com Pivoteamento')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].invert_xaxis()
    axs[0, 0].set_xlabel('Valor do pivô $a_{11}$')
    axs[0, 0].set_ylabel('Erro Relativo Máximo')
    axs[0, 0].set_title('Erro vs Tamanho do Pivô')
    axs[0, 0].legend()
    axs[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    # =================================================================
    # Secção 3: Fatoração LU, múltiplos b
    # =================================================================
    print("--- Secção 3: Fatoração LU (Múltiplos b) ---")
    A2 = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], dtype=float)
    b2_1 = np.array([1, 2, 3], dtype=float)
    b2_2 = np.array([0, 1, -1], dtype=float)

    L, U = fatoracao_lu(A2)
    x_lu_1 = subst_retro(U, subst_prog(L, b2_1))
    x_lu_2 = subst_retro(U, subst_prog(L, b2_2))
    
    print(f"Solução LU b1: {x_lu_1}")
    print(f"Solução LU b2: {x_lu_2}\n")

    # =================================================================
    # Secção 4: Thomas vs. Gauss
    # =================================================================
    print("--- Secção 4: Algoritmo de Thomas ---")
    
    # TODO
    # =================================================================
    # Secção 5: Custo computacional empírico (Cholesky vs LU)
    # =================================================================
    
    print("--- Secção 5: Custo Computacional Empírico (Cholesky vs LU) ---")

    # TODO
    # =================================================================
    # Secção 6: Condicionamento (Matriz de Hilbert)
    # =================================================================
    print("--- Secção 6: Condicionamento (Matriz de Hilbert) ---")

    # TODO
    # =================================================================
    # Secção 7: PageRank
    # =================================================================
    print("--- Secção 7: PageRank Numérico ---")
    G = np.array([
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0]
    ])
    pi_ranks = calcular_pagerank(G, alpha=0.85)
    print(f"Vetor de Ranks: {pi_ranks}")
    
    # =================================================================
    # Secção 8: Desafio (LU Vetorizado)
    # =================================================================
    print("\n--- Secção 8: Desafio (Fatoração LU Vetorizada) ---")
    np.random.seed(42) # Reprodutibilidade
    # Criar uma matriz aleatória com diagonal dominante para evitar pivôs nulos
    A_desafio = np.random.rand(100, 100) + np.eye(100) * 50 
    
    t0 = time.time()
    L_vet, U_vet = fatoracao_lu_vectorized(A_desafio)
    t_vet = time.time() - t0
    
    erro_reconstrucao = np.linalg.norm(L_vet @ U_vet - A_desafio, ord='fro')
    print(f"Matriz 100x100 fatorada em {t_vet:.6f} segundos.")
    print(f"Erro de reconstrução ||LU - A||_F: {erro_reconstrucao:.2e}")
    print("=" * 60)

    # =================================================================
    # Exportar Gráficos
    # =================================================================
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("resultados.pdf", dpi=150)
    print("\n[!] Gráficos gerados com sucesso no ficheiro 'resultados.pdf'.")
    
    # Descomentar a linha abaixo se quiseres ver a janela dos gráficos abrir
    # plt.show()

if __name__ == "__main__":
    main()