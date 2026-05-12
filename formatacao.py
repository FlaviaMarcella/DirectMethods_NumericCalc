"""
Módulo: formatacao.py
Descrição: Funções para formatação visual de saída no console
"""

class Cores:
    """Códigos ANSI para cores no terminal"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Cores principais
    AZUL = '\033[94m'
    VERDE = '\033[92m'
    AMARELO = '\033[93m'
    VERMELHO = '\033[91m'
    CIANO = '\033[96m'
    MAGENTA = '\033[95m'
    BRANCO = '\033[97m'
    
    # Cores de fundo
    BG_AZUL = '\033[104m'
    BG_VERDE = '\033[102m'


def secao_titulo(numero, titulo):
    """Imprime título de seção com formatação"""
    print("\n" + "="*80)
    print(f"{Cores.BOLD}{Cores.AZUL}[SECAO {numero}] {titulo.upper()}{Cores.RESET}")
    print("="*80)


def subsecao(titulo):
    """Imprime subtítulo com formatação"""
    print(f"\n{Cores.BOLD}{Cores.CIANO}>>> {titulo}{Cores.RESET}")
    print(f"  {'-' * 68}")


def destaque_resultado(label, valor, unidade=""):
    """Imprime um resultado em destaque"""
    print(f"  {Cores.VERDE}✓{Cores.RESET} {label}: {Cores.BOLD}{valor}{Cores.RESET} {unidade}".rstrip())


def destaque_erro(label, valor):
    """Imprime um erro em destaque"""
    print(f"  {Cores.VERMELHO}✗{Cores.RESET} {label}: {Cores.BOLD}{valor}{Cores.RESET}")


def destaque_aviso(mensagem):
    """Imprime um aviso"""
    print(f"  {Cores.AMARELO}⚠{Cores.RESET}  {Cores.BOLD}{mensagem}{Cores.RESET}")


def info_matriz(nome, matriz, max_rows=3, max_cols=3):
    """Imprime uma matriz com tamanho limitado e alinhamento"""
    import numpy as np
    if isinstance(matriz, np.ndarray):
        shape = matriz.shape
        print(f"  {Cores.BOLD}{nome} (shape={shape}):{Cores.RESET}")
        
        # Formatar a matriz com melhor alinhamento
        if len(shape) == 1:
            # Vetor
            fmt_str = np.array2string(matriz, formatter={'float_kind': lambda x: f'{x:.4f}'}, 
                                     separator=', ', threshold=np.inf)
            for line in fmt_str.split('\n'):
                print(f"        {line}")
        else:
            # Matriz
            if matriz.shape[0] > max_rows or matriz.shape[1] > max_cols:
                preview = matriz[:max_rows, :max_cols]
                fmt_str = np.array2string(preview, formatter={'float_kind': lambda x: f'{x:.4f}'}, 
                                         separator=', ', threshold=np.inf)
            else:
                fmt_str = np.array2string(matriz, formatter={'float_kind': lambda x: f'{x:.4f}'}, 
                                         separator=', ', threshold=np.inf)
            
            for line in fmt_str.split('\n'):
                print(f"        {line}")
            
            if matriz.shape[0] > max_rows or matriz.shape[1] > max_cols:
                print(f"        {Cores.DIM}... (matriz truncada para visualização){Cores.RESET}")


def comparacao_metodos(resultados_dict):
    """Imprime comparação entre métodos
    
    Args:
        resultados_dict: {'nome_metodo': {'tempo': x, 'residuo': y, ...}}
    """
    print(f"\n  {Cores.BOLD}Comparação de Métodos:{Cores.RESET}")
    for nome, dados in resultados_dict.items():
        print(f"  {Cores.CIANO}→ {nome:40s}{Cores.RESET}", end="")
        if 'tempo' in dados:
            print(f" Tempo: {dados['tempo']:.6f}s", end="")
        if 'residuo' in dados:
            print(f" | Resíduo: {dados['residuo']:.2e}", end="")
        print()


def relatorio_final(concluido=True):
    """Imprime relatório final"""
    if concluido:
        print("\n" + "="*80)
        print(f"{Cores.VERDE}{Cores.BOLD}[OK] INVESTIGACAO CONCLUIDA COM SUCESSO!{Cores.RESET}")
        print(f"{Cores.VERDE}Verifique o arquivo 'resultados.pdf' para visualizar os graficos.{Cores.RESET}")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print(f"{Cores.VERMELHO}{Cores.BOLD}[ERRO] INVESTIGACAO INTERROMPIDA!{Cores.RESET}")
        print("="*80 + "\n")
