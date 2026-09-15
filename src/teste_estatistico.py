import pandas as pd
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# =============================================================================
# 1. FUNÇÕES DE AUDITORIA (Idênticas ao seu TCC)
# =============================================================================
MARGEM_ALUCINACAO = 10 

def contar_itens_edital(texto):
    texto = str(texto)
    return len(re.findall(r'\d+\.\d+(?:\.\d+)*', texto))

def detectar_loop_frasal(texto):
    texto = str(texto)
    frases_cruas = re.split(r'\.\s+', texto)
    frases_limpas = [re.sub(r'\d+', 'X', f).strip() for f in frases_cruas if len(re.sub(r'\d+', 'X', f).strip()) > 15]
    if len(frases_limpas) < 5: return False
    contagem = collections.Counter(frases_limpas)
    for frase, qtd in contagem.items():
        if qtd >= 4: return True
    return False

def detectar_vazamento_prompt(texto):
    texto = str(texto)
    termos_proibidos = ["### Instruction:", "### Response:", "### Input:", "Você é um Revisor Técnico"]
    return any(termo in texto for termo in termos_proibidos)

def extrair_validos(df):
    """Retorna o DataFrame com a coluna booleana 'Valido'"""
    qtd_humano = df['Rascunho_Original'].apply(contar_itens_edital)
    qtd_ia = df['Parecer_Gerado_IA'].apply(contar_itens_edital)
    loop_num = qtd_ia > (qtd_humano + MARGEM_ALUCINACAO)
    loop_lex = df['Parecer_Gerado_IA'].apply(detectar_loop_frasal)
    vaza_pt = df['Parecer_Gerado_IA'].apply(detectar_vazamento_prompt)
    df['Valido'] = ~(loop_num | loop_lex | vaza_pt)
    return df

# =============================================================================
# 2. MOTOR DE DUELOS (McNemar e Wilcoxon)
# =============================================================================
def executar_duelo(nome_ft, file_ft, nome_prompt, file_prompt, cenario):
    print(f"\n{'='*80}")
    print(f"🏆 DUELO {cenario}: AJUSTE FINO vs ENGENHARIA DE PROMPT")
    print(f"🥊 [{nome_ft}]  vs  [{nome_prompt}]")
    print(f"{'='*80}")
    
    try:
        df_ft = pd.read_csv(file_ft, on_bad_lines='skip')
        df_pr = pd.read_csv(file_prompt, on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Erro ao carregar os arquivos: {e}")
        return

    df_ft = extrair_validos(df_ft)
    df_pr = extrair_validos(df_pr)
    
    # --- McNemar (Formato / Textos Válidos) ---
    tabela = pd.crosstab(df_ft['Valido'], df_pr['Valido'])
    for val in [False, True]:
        if val not in tabela.index: tabela.loc[val] = 0
        if val not in tabela.columns: tabela[val] = 0
    tabela = tabela.sort_index(axis=0).sort_index(axis=1)

    p_mcn = mcnemar(tabela, exact=False, correction=True).pvalue
    
    print(f"\n📊 1. TESTE ESTRUTURAL (McNemar - Textos Válidos)")
    print(f"   -> Conformidade {nome_ft}: {df_ft['Valido'].mean():.1%}")
    print(f"   -> Conformidade {nome_prompt}: {df_pr['Valido'].mean():.1%}")
    print(f"   -> Valor-p: {p_mcn:.7f} " + ("(Diferença Significante ⭐)" if p_mcn < 0.05 else "(Empate / Não Significante)"))

    # --- Wilcoxon (Similaridade SBERT) ---
    # Só compara o SBERT onde AMBOS os modelos não alucinaram
    intersecao = df_ft['Valido'] & df_pr['Valido']
    sbert_ft = pd.to_numeric(df_ft.loc[intersecao, 'Similaridade_SBERT'], errors='coerce')
    sbert_pr = pd.to_numeric(df_pr.loc[intersecao, 'Similaridade_SBERT'], errors='coerce')
    
    pares_validos = pd.DataFrame({'FT': sbert_ft, 'PR': sbert_pr}).dropna()

    if len(pares_validos) < 10:
        print("\n⚠️ 2. TESTE SEMÂNTICO (Wilcoxon): Amostra insuficiente.")
    else:
        p_wil = wilcoxon(pares_validos['FT'], pares_validos['PR']).pvalue
        print(f"\n🤖 2. TESTE SEMÂNTICO (Wilcoxon - SBERT em {len(pares_validos)} textos comuns)")
        print(f"   -> Média {nome_ft}: {pares_validos['FT'].mean():.4f}")
        print(f"   -> Média {nome_prompt}: {pares_validos['PR'].mean():.4f}")
        print(f"   -> Valor-p: {p_wil:.7f} " + ("(FT Venceu Significativamente ⭐)" if p_wil < 0.05 and pares_validos['FT'].mean() > pares_validos['PR'].mean() else "(Sem diferença significante)"))

# =============================================================================
# 3. DEFINIÇÃO DOS ARQUIVOS E EXECUÇÃO
# =============================================================================
csv_c1_ft_gemma = "Resultados_modelo_pnld_gemma-3-12b-it-unsloth-bnb-4bit_c1.csv"
csv_c1_pr_gemma = "Resultados_Base_gemma-3-12b-it_4-shot-c1.csv"

csv_c2_ft_gemma = "Resultados_modelo_pnld_gemma-3-12b-it-unsloth-bnb-4bit_c2.csv"
csv_c2_ft_llama = "Resultados_modelo_pnld_Llama-3.2-3B-Instruct-bnb-4bit_c2.csv"
csv_c2_pr_gemma = "Resultados_Base_gemma-3-12b-it_4-shot-c2.csv"

# Duelo 1: Os Campeões do Cenário 1
executar_duelo(
    "Gemma-3-12B Ajustado", csv_c1_ft_gemma,
    "Gemma-3-12B 4-Shot", csv_c1_pr_gemma,
    "CENÁRIO 1"
)

# Duelo 2: Os Campeões do Cenário 2 (Gemma vs Gemma)
executar_duelo(
    "Gemma-3-12B Ajustado", csv_c2_ft_gemma,
    "Gemma-3-12B 4-Shot", csv_c2_pr_gemma,
    "CENÁRIO 2 (Controle de Arquitetura)"
)

# Duelo 3: Pico Semântico C2 (Llama 3B vs Gemma 12B Prompt)
executar_duelo(
    "Llama-3.2-3B Ajustado", csv_c2_ft_llama,
    "Gemma-3-12B 4-Shot", csv_c2_pr_gemma,
    "CENÁRIO 2 (Pico Semântico vs Melhor Prompt)"
)

# =============================================================================
# 4. GERAÇÃO DO BOXPLOT COMPARATIVO (Nomes Completos)
# =============================================================================
print(f"\n🎨 Gerando Boxplot dos Campeões...")

# AQUI: Usando \n para empilhar Nome do Modelo na 1ª linha e Técnica/Cenário na 2ª
modelos_plot = [
    {"nome": "Gemma-3-12B\nAjustado (C1)", "file": csv_c1_ft_gemma},
    {"nome": "Gemma-3-12B\n4-Shot (C1)",   "file": csv_c1_pr_gemma},
    {"nome": "Gemma-3-12B\nAjustado (C2)", "file": csv_c2_ft_gemma},
    {"nome": "Llama-3.2-3B\nAjustado (C2)", "file": csv_c2_ft_llama},
    {"nome": "Gemma-3-12B\n4-Shot (C2)",   "file": csv_c2_pr_gemma}
]

dados_boxplot = []
for mod in modelos_plot:
    try:
        df = pd.read_csv(mod['file'], on_bad_lines='skip')
        df = extrair_validos(df)
        scores = pd.to_numeric(df.loc[df['Valido'], 'Similaridade_SBERT'], errors='coerce').dropna().tolist()
        for s in scores:
            dados_boxplot.append({"Configuração": mod['nome'], "SBERT": s})
    except:
        pass

if dados_boxplot:
    df_plot = pd.DataFrame(dados_boxplot)
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    ax = sns.boxplot(x="Configuração", y="SBERT", hue="Configuração", data=df_plot, palette="Set2", width=0.6, legend=False)
    sns.stripplot(x="Configuração", y="SBERT", hue="Configuração", data=df_plot, palette="dark:black", alpha=0.3, size=3, legend=False)

    plt.title("Comparativo de Similaridade SBERT: Fine-Tuning vs Prompt Engineering", fontsize=14, pad=15)
    plt.ylabel("Score SBERT", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.ylim(0.4, 1.0)
    
    # Adiciona espaçamento extra na base da imagem
    plt.subplots_adjust(bottom=0.15)
    
    # Salva garantindo que a "caixa delimitadora" envolva tudo (bbox_inches='tight')
    plt.savefig("boxplot_ft_vs_prompt.pdf", dpi=300, bbox_inches='tight')
    print(f"✅ Boxplot salvo como 'boxplot_ft_vs_prompt.pdf' com nomes completos!")