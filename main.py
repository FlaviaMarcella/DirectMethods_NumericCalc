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
from gauss import resolver_gauss, resolver_gauss_sem_pivoteamento, resolver_gauss_precision, experimento_estabilidade_gauss
from lu import fatoracao_lu, fatoracao_lu_vectorized, experimento_desempenho_lu
from pagerank import calcular_pagerank

def main():
    # Garantir reprodutibilidade global
    np.random.seed(42)

    # Setup da figura para resultados.pdf (Grelha 4x2 para os 7 gráficos)
    fig, axs = plt.subplots(4, 2, figsize=(14, 22))
    fig.suptitle('Plano de Investigação: Sistemas Lineares - Resultados Consolidados', fontsize=18, fontweight='bold')

    # =========================================================================
    # Seção 1: Verificação básica
    # =========================================================================
    print("--- Seção 1: Verificação básica ---")
    A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=np.float32)
    b = np.array([1, 2, 3], dtype=np.float32)
    x = resolver_gauss(A, b)
    residuo = np.linalg.norm(A @ x - b)
    print(f"    Seção 1 Matriz: {A}")
    print(f"    Seção 1 Vetor b: {b}")
    print(f"    Seção 1 Resíduo: {residuo:.2e}\n")
    print(f"    Seção 1 Solução: {x}")

    # Verificando a solução: A @x deve ser aproximadamente b
    print(f"    Seção 1 Verificação: (Ax) = {A @ x}")
    print(f"    Seção 1 Diferença: {A @ x - b}")


    # =========================================================================
    # Seção 1.3: Efeito do pivoteamento
    # =========================================================================

    print('--- Efeito do pivoteamento na precisão ---')

    # Defina a matriz A e o vetor b para um sistema de equações lineares
    A = np.array([
    [0.0003, 3],
    [1, 1]
    ], dtype=np.float32)

    b = np.array([2.0001, 1], dtype=np.float32)

    # Resolva o sistema usando a função resolver_gauss_sem_pivoteamento
    x_solucao_sem_pivoteamento = resolver_gauss_sem_pivoteamento(A, b)
    x_solucao_com_pivoteamento = resolver_gauss(A, b)

    # Resíduo -> ∥Ax − b∥2
    residuo_sem_pivot =  np.linalg.norm(A @ x_solucao_sem_pivoteamento - b)
    residuo_com_pivot =  np.linalg.norm(A @ x_solucao_com_pivoteamento - b)

    print(f"Matriz A:\n{A}")
    print(f"Vetor b: {b}")

    print(f"Residuo sem Pivot: {residuo_sem_pivot}")
    print(f"Residuo c/Pivot: {residuo_com_pivot}")

    print(f"Solução x sem pivoteamento: {x_solucao_sem_pivoteamento}")
    print(f"Solução x com pivoteamento: {x_solucao_com_pivoteamento}")

    # Calcular os valores verdadeiros de x1 e x2 para comparação
    x1_true = 1/3
    x2_true = 2/3
    x_true = np.array([x1_true, x2_true])

    # Calcular o erro relativo sem pivoteamento
    erro_relativo_sem_pivoteamento_x1 = np.abs(x_solucao_sem_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
    erro_relativo_sem_pivoteamento_x2 = np.abs(x_solucao_sem_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])

    # Calcular o erro relativo com pivoteamento
    erro_relativo_com_pivoteamento_x1 = np.abs(x_solucao_com_pivoteamento[0] - x_true[0]) / np.abs(x_true[0])
    erro_relativo_com_pivoteamento_x2 = np.abs(x_solucao_com_pivoteamento[1] - x_true[1]) / np.abs(x_true[1])

    print(f"Erro Relativo Sem Pivoteamento")
    print(f"Erro Relativo X1: {erro_relativo_sem_pivoteamento_x1}")
    print(f"Erro Relativo X2: {erro_relativo_sem_pivoteamento_x2}")

    print(f"Erro Relativo Com Pivoteamento")
    print(f"Erro Relativo X1: {erro_relativo_com_pivoteamento_x1}")
    print(f"Erro Relativo X2: {erro_relativo_com_pivoteamento_x2}")

    print("--- Seção 2: Efeito do pivoteamento---")



    a11_vals = [0.1, 1e-3, 1e-6, 1e-9]
    erro_sem, erro_com = experimento_estabilidade_gauss(a11_vals)

    # Plotar exatamente como no notebook: dois log-log subplots e salvar
    fig_q1, ax_q1 = plt.subplots(1, 2, figsize=(10, 6))
    ax_q1[0].loglog(a11_vals, erro_sem, 'o-', label='Sem Pivoteamento', color='blue')
    ax_q1[0].loglog(a11_vals, erro_com, 'x-', label='Com Pivoteamento', color='red')
    ax_q1[0].set_xlabel('Valor de a11 (Log Scale)')
    ax_q1[0].set_ylabel('Erro Relativo em x1 (Log Scale)')
    ax_q1[0].set_title('Erro Relativo em x1 vs. a11')
    ax_q1[0].legend()
    ax_q1[0].grid(True, which='both', ls='-', alpha=0.7)

    ax_q1[1].loglog(a11_vals, erro_sem, 'o-', label='Sem Pivoteamento', color='blue')
    ax_q1[1].loglog(a11_vals, erro_com, 'x-', label='Com Pivoteamento', color='red')
    ax_q1[1].set_xlabel('Valor de a11 (Log Scale)')
    ax_q1[1].set_ylabel('Erro Relativo em x1 (Log Scale)')
    ax_q1[1].set_title('Erro Relativo em x1 vs. a11 (repetido)')
    ax_q1[1].legend()
    ax_q1[1].grid(True, which='both', ls='-', alpha=0.7)

    plt.tight_layout()
    plt.savefig('q1_4_estabilidade.pdf', dpi=150, bbox_inches='tight')
    plt.close()

    # também inserir um resumo no painel consolidado
    axs[0, 0].loglog(a11_vals, erro_sem, 'o-', color='blue')
    axs[0, 0].loglog(a11_vals, erro_com, 'x-', color='red')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel(r'Valor do pivô $a_{11}$')
    axs[0, 0].set_ylabel(r'Erro Relativo em $x_1$')
    axs[0, 0].set_title('Q1.4: Estabilidade vs Tamanho do Pivô')
    axs[0, 0].legend(['Sem Pivoteamento','Com Pivoteamento'])
    axs[0, 0].grid(True, which="both", ls="-", alpha=0.2)
    # =========================================================================
    # Seção 3: Fatoração LU (Gráfico Q3.2)
    # =========================================================================
    print("--- Seção 3: Fatoração LU vs Gauss Repetido (Q3.2) ---")
    ns = [50, 100, 150, 200]
    t_gauss_total, t_lu_total = experimento_desempenho_lu(ns)

    axs[0, 1].bar(['Gauss (50x)', 'LU (1x) + 50 Subst.'], [t_gauss_total, t_lu_total], color=['#e74c3c', '#2ecc71'])
    axs[0, 1].set_ylabel('Tempo Total (segundos)')
    axs[0, 1].set_title('Q3.2: Eficiência com Múltiplos b')
    for i, v in enumerate([t_gauss_total, t_lu_total]):
        axs[0, 1].text(i, v + 0.01, f"{v:.4f}s", ha='center', fontweight='bold')

    # =========================================================================
    # Seção 4: Thomas vs. Gauss (Gráfico Q4.2) - TODO (Colega)
    # =========================================================================
    print("--- Seção 4: Thomas vs Gauss (Q4.2) --- [TODO: Colega]")
    axs[1, 0].text(0.5, 0.5, "Tópico 4: Thomas vs Gauss\n(Parte do Colega)", ha='center', va='center')
    axs[1, 0].set_title('Q4.2: Thomas vs Gauss')

    # =========================================================================
    # Seção 5: Custo computacional (Gráfico Q5.2) - TODO (Colega)
    # =========================================================================
    print("--- Seção 5: Cholesky vs LU (Q5.2) --- [TODO: Colega]")
    axs[1, 1].text(0.5, 0.5, "Tópico 5: Cholesky vs LU\n(Parte do Colega)", ha='center', va='center')
    axs[1, 1].set_title('Q5.2: Cholesky vs LU')

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
    # Matriz de adjacência do exemplo (G[i,j]=1 se j->i)
    G_mat = np.array([
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0]
    ])
    pi_vector = calcular_pagerank(G_mat)
    paginas = ['P1', 'P2', 'P3', 'P4']
    axs[2, 1].bar(paginas, pi_vector, color='gold', edgecolor='black')
    axs[2, 1].set_ylabel(r'Importância ($\pi$)')
    axs[2, 1].set_title('Q7.3: Distribuição de PageRank')
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
        
        # Simular versão lenta (sem vetorização) para o gráfico se necessário, 
        # mas aqui comparamos a fatoração LU padrão vs vetorizada.
        t0 = time.time()
        fatoracao_lu(Av)
        t_loop.append(time.time()-t0)
        
        t0 = time.time()
        fatoracao_lu_vectorized(Av)
        t_vec.append(time.time()-t0)

    axs[3, 0].plot(n_v, t_loop, 'k--', marker='o', label='LU (Loops Python)')
    axs[3, 0].plot(n_v, t_vec, 'c-', marker='s', label='LU (Vetorizado NumPy)')
    axs[3, 0].set_xlabel('Tamanho da Matriz (n)')
    axs[3, 0].set_ylabel('Tempo (s)')
    axs[3, 0].set_title('Q8.1: Benefício da Vetorização')
    axs[3, 0].legend()
    axs[3, 0].grid(True)
    
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
