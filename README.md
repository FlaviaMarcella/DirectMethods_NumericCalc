# Métodos Diretos para Cálculo Numérico

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.7+-blue)

Implementação completa de métodos diretos para resolução de sistemas lineares, desenvolvida como projeto acadêmico da disciplina **Cálculo Numérico**. O projeto explora técnicas fundamentais de álgebra linear numérica com análise de precisão, estabilidade e desempenho.

## 📋 Conteúdo

- [Sobre](#sobre)
- [Métodos Implementados](#métodos-implementados)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Resultados](#resultados)
- [Requisitos](#requisitos)

## Sobre

Este projeto implementa e compara diversos métodos diretos para a resolução de sistemas lineares da forma **Ax = b**. Inclui testes de estabilidade numérica, análise de pivoteamento, e aplicações práticas como o algoritmo PageRank.

### Objetivos

- ✅ Implementar métodos diretos clássicos de álgebra linear numérica
- ✅ Analisar efeitos de precisão e pivoteamento
- ✅ Comparar desempenho entre diferentes abordagens
- ✅ Aplicar os métodos em problemas reais (PageRank)
- ✅ Gerar visualizações dos resultados

## Métodos Implementados

### 1. **Eliminação de Gauss** (`gauss.py`)
Implementação da clássica Eliminação de Gauss com:
- Versão sem pivoteamento
- Versão com pivoteamento parcial
- Detecção de singularidade
- Suporte para precisão float32 e float64

### 2. **Fatoração LU** (`lu.py`)
Método de Doolittle para decomposição LU com:
- Substituição progressiva (forward substitution)
- Substituição retroativa (backward substitution)
- Versão vetorizada otimizada
- Ideal para sistemas com múltiplos vetores de carga

### 3. **Fatoração de Cholesky** (`cholesky.py`)
Especializada para matrizes simétricas positivas definidas (SPD):
- Decomposição A = L × L^T
- Validação automática de propriedades
- Ideal para problemas de mínimos quadrados

### 4. **Algoritmo de Thomas (TDMA)** (`thomas.py`)
Método otimizado O(n) para sistemas tridiagonais:
- Complexidade linear
- Ideal para problemas de diferenças finitas
- Aplicações em EDP (equações diferenciais parciais)

### 5. **Número de Condição** (`condicao.py`)
Análise de estabilidade numérica:
- Cálculo do número de condição
- Estimativas de erro
- Identificação de sistemas mal-condicionados

### 6. **PageRank** (`pagerank.py`)
Aplicação prática do algoritmo PageRank do Google:
- Implementado como sistema linear
- Utiliza Eliminação de Gauss
- Análise de relevância em grafos

## Instalação

### Pré-requisitos
- Python 3.7 ou superior
- pip

### Passos

1. Clone o repositório:
```bash
git clone https://github.com/FlaviaMarcella/DirectMethods---NumericCalc.git
cd DirectMethods---NumericCalc
```

2. Crie e ative um ambiente virtual:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Executar o programa principal
```bash
python main.py
```

Isso executará todos os testes implementados e gerará um arquivo `resultados.pdf` com gráficos e visualizações.

### Usar módulos individuais

#### Eliminação de Gauss:
```python
from gauss import resolver_gauss
import numpy as np

A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=float)
b = np.array([1, 2, 3], dtype=float)
x = resolver_gauss(A, b)
print(f"Solução: {x}")
```

#### Fatoração LU:
```python
from lu import fatoracao_lu, subst_prog, subst_retro
import numpy as np

A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

L, U = fatoracao_lu(A)
y = subst_prog(L, b)
x = subst_retro(U, y)
print(f"Solução: {x}")
```

#### Fatoração de Cholesky:
```python
from cholesky import resolver_cholesky
import numpy as np

A = np.array([[4, 2], [2, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
x = resolver_cholesky(A, b)
print(f"Solução: {x}")
```

#### PageRank:
```python
from pagerank import calcular_pagerank
import numpy as np

G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
ranks = calcular_pagerank(G, alpha=0.85)
print(f"PageRanks: {ranks}")
```

## Estrutura do Projeto

```
DirectMethods---NumericCalc/
├── main.py                 # Script principal - executa todos os testes
├── gauss.py               # Eliminação de Gauss
├── lu.py                  # Fatoração LU
├── cholesky.py            # Fatoração de Cholesky
├── thomas.py              # Algoritmo de Thomas (TDMA)
├── condicao.py            # Análise de número de condição
├── pagerank.py            # Algoritmo PageRank
├── resultados.pdf         # Saída com gráficos e resultados
├── requirements.txt       # Dependências do projeto
├── .gitignore            # Configuração Git
└── README.md             # Este arquivo
```

## Resultados

Ao executar `python main.py`, o programa:

1. ✅ Testa cada método com matrizes de diferentes tamanhos
2. ✅ Analisa o efeito do pivoteamento na estabilidade numérica
3. ✅ Compara precisão (erro residual) entre métodos
4. ✅ Mede tempo de execução para análise de desempenho
5. ✅ Calcula o PageRank de um grafo de exemplo
6. ✅ Gera gráficos comparativos salvos em `resultados.pdf`

### Exemplo de Saída:
```
--- Secção 1: Verificação Básica ---
Secção 1 Solução: [...]
Secção 1 Resíduo: 2.22e-16

--- Secção 2: Efeito do Pivoteamento (float32) ---
Solução SEM Pivoteamento: [...]
Solução COM Pivoteamento: [...]
```

## Requisitos

```
numpy>=1.19.0
matplotlib>=3.3.0
scipy>=1.5.0
```

Instale com:
```bash
pip install -r requirements.txt
```

## Notas Acadêmicas

Este projeto foi desenvolvido para compreender:

- 📐 Conceitos fundamentais de álgebra linear numérica
- 🎯 Trade-offs entre estabilidade e desempenho
- 💻 Implementação eficiente de algoritmos numéricos
- 📊 Análise de precisão e erros numéricos
- 🔗 Aplicações práticas em computação científica

## Licença

Este projeto está sob a licença **MIT**. Veja o arquivo `LICENSE` para mais detalhes.

## Autor

Desenvolvido por **Flavia Marcella** como projeto acadêmico.

## Contribuições

Sugestões e melhorias são bem-vindas! Sinta-se à vontade para:

- 🐛 Reportar bugs
- 💡 Sugerir melhorias
- 📝 Enviar pull requests
- 📧 Entrar em contato com feedback

## Referências

- Golub, G. H., & Van Loan, C. F. (2013). Matrix computations (4th ed.)
- Demmel, J. W. (1997). Applied numerical linear algebra
- Trefethen, L. N., & Bau III, D. (1997). Numerical linear algebra

---

**Última atualização:** Maio de 2026
