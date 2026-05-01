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
from lu import fatoracao_lu, subst_prog, subst_retro, fatoracao_lu_vectorized, resolver_lu
from cholesky import resolver_cholesky

from pagerank import calcular_pagerank

def main():
    # Setup da figura para resultados.pdf (Grelha 4x2 para os 7 gráficos)
    fig, axs = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle('Plano de Investigação: Sistemas Lineares - Resultados Consolidados', fontsize=18, fontweight='bold')

    # =========================================================================
    # Seção 1: Verificação básica
    # =========================================================================
    print("--- Seção 1: Verificação básica ---")
    A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    x = resolver_gauss(A, b)
    print(f"Seção 1 Solução: {x}")
    print(f"Seção 1 Resíduo: {np.linalg.norm(A @ x - b):.2e}\n")

    # =========================================================================
    # TODO: Seção 2 Efeito do pivoteamento (Gráfico Q1.4)
    # =========================================================================
    print("--- Seção 2: Efeito do pivoteamento (Q1.4) ---")
    a11_vals = [10**-1, 10**-3, 10**-6, 10**-9]
    erro_sem, erro_com = [], []
    # Solução exata teórica para comparação
    x_true = np.array([1/3, 2/3], dtype=np.float32)

    for a11 in a11_vals:
        A_p = np.array([[a11, 3], [1, 1]], dtype=np.float32)
        b_p = np.array([2.0001, 1], dtype=np.float32)
        
        # Execução garantindo float32 (essencial para ver o erro explodir)
        xs = resolver_gauss_sem(A_p.copy(), b_p.copy())
        xc = resolver_gauss(A_p.copy(), b_p.copy())
        
        # Erro relativo em x1 (conforme Q1.4 do notebook)
        erro_sem.append(abs(xs[0] - x_true[0]) / x_true[0])
        erro_com.append(abs(xc[0] - x_true[0]) / x_true[0])

    axs[0, 0].plot(a11_vals, erro_sem, 'ro-', label='Sem Pivoteamento')
    axs[0, 0].plot(a11_vals, erro_com, 'bs--', label='Com Pivoteamento')
    axs[0, 0].set_xscale('log'); axs[0, 0].set_yscale('log'); axs[0, 0].invert_xaxis()
    axs[0, 0].set_xlabel('Valor do pivô $a_{11}$'); axs[0, 0].set_ylabel('Erro Relativo em $x_1$')
    axs[0, 0].set_title('Q1.4: Estabilidade vs Tamanho do Pivô')
    axs[0, 0].legend(); axs[0, 0].grid(True, which="both", ls="-", alpha=0.2)

    # =========================================================================
    # TODO: Seção 3 Fatoração LU (Gráfico Q3.2)
    # =========================================================================
    print("--- Seção 3: Fatoração LU vs Gauss Repetido (Q3.2) ---")
    n_c = 200; n_b = 50
    Ac = np.random.rand(n_c, n_c).astype(np.float32) + np.eye(n_c) * n_c
    bs = [np.random.rand(n_c).astype(np.float32) for _ in range(n_b)]

    t0 = time.time()
    for bi in bs: resolver_gauss(Ac, bi)
    t_gauss_total = time.time() - t0

    t0 = time.time()
    L, U = fatoracao_lu(Ac)
    for bi in bs: subst_retro(U, subst_prog(L, bi))
    t_lu_total = time.time() - t0

    axs[0, 1].bar(['Gauss (50x)', 'LU (1x) + 50 Subst.'], [t_gauss_total, t_lu_total], color=['#e74c3c', '#2ecc71'])
    axs[0, 1].set_ylabel('Tempo Total (segundos)'); axs[0, 1].set_title('Q3.2: Eficiência com Múltiplos b')
    for i, v in enumerate([t_gauss_total, t_lu_total]):
        axs[0, 1].text(i, v + 0.01, f"{v:.4f}s", ha='center', fontweight='bold')

    # =========================================================================
    # TODO: Seção 4 Thomas vs. Gauss (Gráfico Q4.2)
    # =========================================================================
    print("--- Seção 4: Thomas vs Gauss (Q4.2) ---")

    # =========================================================================
    # TODO: Seção 5 Custo computacional (Gráfico Q5.2)
    # =========================================================================
    print("--- Seção 5: Cholesky vs LU (Q5.2) ---")

    # =========================================================================
    # TODO: Seção 6 Condicionamento (Gráfico Q6.1)
    # =========================================================================
    print("--- Seção 6: Condicionamento (Q6.1) ---")


    # =========================================================================
    # TODO: Seção 7 PageRank (Gráfico Q7.3)
    # =========================================================================
    print("--- Seção 7: PageRank (Q7.3) ---")
    G_mat = np.array([[0,0,1,1], [1,0,0,0], [1,1,0,1], [1,1,0,0]])
    pi_vector = calcular_pagerank(G_mat)
    paginas = ['P1', 'P2', 'P3', 'P4']
    axs[2, 1].bar(paginas, pi_vector, color='gold', edgecolor='black')
    axs[2, 1].set_ylabel('Importância ($\pi$)'); axs[2, 1].set_title('Q7.3: Distribuição de PageRank')
    for i, v in enumerate(pi_vector):
        axs[2, 1].text(i, v + 0.005, f"{v:.3f}", ha='center')

    # =========================================================================
    # Seção 8: Impacto da Vetorização (Gráfico Q8.1)
    # =========================================================================
    print("--- Seção 8: Impacto da Vetorização (Q8.1) ---")
    n_v = [50, 100, 150, 200]
    t_loop, t_vec = [], []
    for n in n_v:
        Av = np.random.rand(n,n).astype(np.float32) + np.eye(n)*n
        t0 = time.time(); fatoracao_lu(Av); t_loop.append(time.time()-t0)
        t0 = time.time(); fatoracao_lu_vectorized(Av); t_vec.append(time.time()-t0)

    axs[3, 0].plot(n_v, t_loop, 'k--', marker='o', label='LU (Loops Python)')
    axs[3, 0].plot(n_v, t_vec, 'c-', marker='s', label='LU (Vetorizado NumPy)')
    axs[3, 0].set_xlabel('Tamanho da Matriz (n)'); axs[3, 0].set_ylabel('Tempo (s)')
    axs[3, 0].set_title('Q8.1: Benefício da Vetorização'); axs[3, 0].legend(); axs[3, 0].grid(True)
    
    # Eixo 3,1 fica vazio ou para anotações
    axs[3, 1].axis('off')
    axs[3, 1].text(0.1, 0.5, "Relatório Completo\nCálculo Numérico 2026\nUNIFAL-MG", 
                   fontsize=12, style='italic', bbox={'facecolor': 'grey', 'alpha': 0.1, 'pad': 10})

    # Salvando resultados
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("resultados.pdf", dpi=150)
    print("\n[!] INVESTIGAÇÃO CONCLUÍDA. Verifique o arquivo 'resultados.pdf'.")

if __name__ == "__main__":
    main()