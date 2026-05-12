"""
Módulo: condicao.py
Descrição: Funções para análise de condicionamento, matriz de Hilbert e 
estabilidade numérica em sistemas lineares.
"""

import numpy as np
from scipy.linalg import hilbert

def experimento_hilbert(n):
    """Resolve Hn x = b e retorna o número de condicionamento e erro relativo."""
    H = hilbert(n)
    x_exato = np.ones(n)
    b = H @ x_exato
    
    # Resolve usando o solver padrão
    x_calc = np.linalg.solve(H, b)
    
    # Cálculo do erro relativo e kappa
    erro_rel = np.linalg.norm(x_calc - x_exato) / np.linalg.norm(x_exato)
    kappa = np.linalg.cond(H, p=2)
    
    return kappa, erro_rel

def perturbar_b(A, b, nivel=1e-6, n_amostras=100):
    """Mede a amplificação do erro ao perturbar o vetor b."""
    x_exato = np.linalg.solve(A, b)
    amplificacoes = []
    
    for _ in range(n_amostras):
        db = nivel * np.random.randn(len(b))
        x_pert = np.linalg.solve(A, b + db)
        
        erro_rel_x = np.linalg.norm(x_pert - x_exato) / np.linalg.norm(x_exato)
        erro_rel_b = np.linalg.norm(db) / np.linalg.norm(b)
        
        amplificacoes.append(erro_rel_x / erro_rel_b)
        
    return np.array(amplificacoes)