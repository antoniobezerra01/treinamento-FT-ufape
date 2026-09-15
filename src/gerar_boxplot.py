import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

# Configuração de estilo acadêmico para os gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14, 'pdf.fonttype': 42}) # Fonte um pouco maior para ficar perfeito no PDF

def contar_itens_edital(texto):
    return len(re.findall(r'\d+\.\d+(?:\.\d+)*', str(texto)))

def higienizar_dataframe(df):
    MARGEM_ALUCINACAO = 10
    df['Qtd_Humano'] = df['Rascunho_Original'].apply(contar_itens_edital)
    df['Qtd_IA'] = df['Parecer_Gerado_IA'].apply(contar_itens_edital)
    loop_num = df['Qtd_IA'] > (df['Qtd_Humano'] + MARGEM_ALUCINACAO)
    vazamento = df['Parecer_Gerado_IA'].apply(lambda x: any(t in str(x) for t in ["### Instruction:", "### Response:", "Você é um Revisor Técnico"]))
    df['Falha'] = loop_num | vazamento
    return df[df['Falha'] == False].copy()

arquivos = glob.glob("**/*Resultados_*.csv", recursive=True)
dados_grafico = []

for arquivo in arquivos:
    try:
        df = pd.read_csv(arquivo, on_bad_lines='skip')
        if 'Similaridade_SBERT' not in df.columns: continue
        
        df_limpo = higienizar_dataframe(df)
        if df_limpo.empty: continue
            
        nome_arquivo = os.path.basename(arquivo)
        cenario = "C1" if "_c1" in nome_arquivo.lower() or "-c1" in nome_arquivo.lower() else "C2"
            
        is_ajustado = "modelo_pnld" in nome_arquivo
        tecnica = "Fine-Tuning" if is_ajustado else "Prompt"
        
        estrategia = "Ajustado"
        if not is_ajustado:
            match = re.search(r'(\d+)[-_]?shot', nome_arquivo, re.IGNORECASE)
            estrategia = f"{match.group(1)}-Shot" if match else "0-Shot"
            
        modelo_base = nome_arquivo.replace("Resultados_", "").replace("Base_", "").replace("modelo_pnld_", "")
        modelo_base = re.sub(r'\d+[-_]?shot[-_]?', '', modelo_base, flags=re.IGNORECASE)
        remover_tags = [".csv", "_c1", "-c1", "_c2", "-c2", "-it-unsloth-bnb-4bit", "-it-bnb-4bit", "-bnb-4bit", "Meta-", "-Instruct", "-it"]
        for tag in remover_tags:
            modelo_base = modelo_base.replace(tag, "")
        modelo_base = modelo_base.strip("_ -")
        
        # --- A MÁGICA ACONTECE AQUI: Padronização exata dos nomes para o Top 3 ---
        if "gemma-3-12b" in modelo_base.lower(): modelo_base = "Gemma-3-12B"
        elif "gemma-3-4b" in modelo_base.lower(): modelo_base = "Gemma-3-4B"
        elif "llama-3.2-3b" in modelo_base.lower(): modelo_base = "Llama-3.2-3B"
        else: continue # Se não for um dos Top 3, IGNORA e não coloca no gráfico
        
        for _, row in df_limpo.iterrows():
            dados_grafico.append({
                'Cenário': cenario,
                'Modelo': modelo_base,
                'Técnica': tecnica,
                'Estratégia': estrategia,
                'SBERT': pd.to_numeric(row.get('Similaridade_SBERT', np.nan), errors='coerce'),
                'Cosseno': pd.to_numeric(row.get('Similaridade_Cosseno', np.nan), errors='coerce')
            })
    except Exception as e:
        print(f"Erro em {arquivo}: {e}")

df_master = pd.DataFrame(dados_grafico).dropna()
cores_tecnica = {"Fine-Tuning": "#2ecc71", "Prompt": "#3498db"}

# Define a ordem exata de exibição no eixo X (Do melhor pro "pior" do Top 3)
ordem_modelos = ["Gemma-3-12B", "Llama-3.2-3B", "Gemma-3-4B"]

def gerar_graficos_top3(cenario_alvo):
    df_cenario = df_master[df_master['Cenário'] == cenario_alvo]
    if df_cenario.empty: return
    
    # 1. Boxplot SBERT (Ajustado vs 4-Shot)
    plt.figure(figsize=(9, 6))
    df_plot1 = df_cenario[df_cenario['Estratégia'].isin(['Ajustado', '4-Shot'])]
    sns.boxplot(data=df_plot1, x='Modelo', y='SBERT', hue='Técnica', palette=cores_tecnica, order=ordem_modelos, showfliers=False)
    plt.title(f"Top 3 Modelos: Similaridade Semântica (SBERT) - {cenario_alvo}", pad=15)
    plt.ylabel("Escore SBERT")
    plt.xlabel("") # Remove a palavra "Modelo" pois já fica óbvio
    plt.legend(title="", loc="lower right") # Legenda limpa no canto
    plt.tight_layout()
    plt.savefig(f"grafico_top3_sbert_{cenario_alvo}.pdf")
    plt.close()

    # 2. Boxplot Cosseno (Ajustado vs 4-Shot)
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df_plot1, x='Modelo', y='Cosseno', hue='Técnica', palette=cores_tecnica, order=ordem_modelos, showfliers=False)
    plt.title(f"Top 3 Modelos: Adesão Lexical Estrita (Cosseno) - {cenario_alvo}", pad=15)
    plt.ylabel("Escore Cosseno")
    plt.xlabel("")
    plt.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.savefig(f"grafico_top3_cosseno_{cenario_alvo}.pdf")
    plt.close()

# Executa para os dois cenários
gerar_graficos_top3("C1")
gerar_graficos_top3("C2")
print("✅ Gráficos do Top-3 gerados com sucesso e separados por cenário!")