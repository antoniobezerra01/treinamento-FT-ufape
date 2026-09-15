import pandas as pd
import numpy as np
import glob
import os
import re
import collections

# =============================================================================
# 1. FUNÇÕES DE HIGIENIZAÇÃO (A MALHA FINA)
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

# =============================================================================
# 2. CONFIGURAÇÕES E BUSCA DE ARQUIVOS
# =============================================================================
print("📂 Procurando arquivos CSV de resultados...")

# Busca profunda pegando tanto os Ajustados quanto os de Base
arquivos = glob.glob("**/*Resultados_*.csv", recursive=True)
if not arquivos:
    todos_csvs = glob.glob("**/*.csv", recursive=True)
    arquivos = [f for f in todos_csvs if "Resultados" in os.path.basename(f)]

metricas = {
    'Similaridade_Cosseno': 'Cosseno',
    'Similaridade_Jaccard': 'Jaccard',
    'Similaridade_SpaCy': 'SpaCy',
    'Similaridade_SBERT': 'SBERT'
}

dados_tabela = []

# =============================================================================
# 3. PROCESSAMENTO E CÁLCULO DAS MÉDIAS
# =============================================================================
for arquivo in arquivos:
    try:
        df = pd.read_csv(arquivo, on_bad_lines='skip')
        
        # Trava de segurança
        if 'Rascunho_Original' not in df.columns or 'Parecer_Gerado_IA' not in df.columns:
            continue
            
        total_textos = len(df)
        
        # --- APLICA A HIGIENIZAÇÃO (FILTRO DE ALUCINAÇÕES) ---
        df['Qtd_Humano'] = df['Rascunho_Original'].apply(contar_itens_edital)
        df['Qtd_IA'] = df['Parecer_Gerado_IA'].apply(contar_itens_edital)
        df['Loop_Numerico'] = df['Qtd_IA'] > (df['Qtd_Humano'] + MARGEM_ALUCINACAO)
        df['Loop_Lexical'] = df['Parecer_Gerado_IA'].apply(detectar_loop_frasal)
        df['Vazamento'] = df['Parecer_Gerado_IA'].apply(detectar_vazamento_prompt)
        
        # Identifica os textos válidos
        df['Falha_Severa'] = df['Loop_Numerico'] | df['Loop_Lexical'] | df['Vazamento']
        df_valido = df[df['Falha_Severa'] == False].copy()
        
        textos_validos = len(df_valido)
        
        # --- FORMATAÇÃO DO NOME DO MODELO ---
        nome_bruto = os.path.basename(arquivo).replace(".csv", "")
        
        # Extrai o Cenário
        cenario = "??"
        if nome_bruto.lower().endswith(("_c1", "-c1")):
            cenario = "C1"
            nome_bruto = nome_bruto[:-3]
        elif nome_bruto.lower().endswith(("_c2", "-c2")):
            cenario = "C2"
            nome_bruto = nome_bruto[:-3]
            
        # Extrai a Tag de Treino e Limpa as palavras feias (estética da tabela)
        if "modelo_pnld_" in nome_bruto:
            nome_bruto = nome_bruto.replace("Resultados_modelo_pnld_", "[AJUSTADO] ")
        elif "Resultados_Base_" in nome_bruto:
            nome_bruto = nome_bruto.replace("Resultados_Base_", "")
            match_shot = re.search(r'(\d+)[-_]?shot', nome_bruto, re.IGNORECASE)
            if match_shot:
                num_shots = match_shot.group(1)
                nome_sem_shot = re.sub(r'[-_]?\d+[-_]?shot', '', nome_bruto, flags=re.IGNORECASE).strip("_ ")
                nome_bruto = f"[{num_shots}-SHOT]   {nome_sem_shot}"
            else:
                nome_bruto = f"[0-SHOT]   {nome_bruto}"
                
        # Limpeza fina para deixar o nome elegante
        nome_limpo = nome_bruto.replace("-it-unsloth-bnb-4bit", "")\
                               .replace("-it-bnb-4bit", "")\
                               .replace("Meta-", "")\
                               .replace("-Instruct", "")\
                               .replace("-it", "")
                               
        nome_final = f"[{cenario}] {nome_limpo}"
        
        # Inicializa a linha da tabela
        linha = {
            'Configuração / Modelo': nome_final,
            'Total Avaliado': total_textos,
            'Válidos (Usados na Média)': textos_validos
        }
        
        # --- CÁLCULO DAS MÉDIAS SOBRE A BASE LIMPA ---
        for col_orig, col_nome in metricas.items():
            if col_orig in df_valido.columns:
                val = pd.to_numeric(df_valido[col_orig], errors='coerce').dropna()
                if len(val) > 0:
                    linha[f'{col_nome} (Média)'] = val.mean()
                    linha[f'{col_nome} (Mediana)'] = val.median()
                    linha[f'{col_nome} (DP)'] = val.std()
                else:
                    linha[f'{col_nome} (Média)'] = np.nan
                    linha[f'{col_nome} (Mediana)'] = np.nan
                    linha[f'{col_nome} (DP)'] = np.nan
            else:
                linha[f'{col_nome} (Média)'] = np.nan
                linha[f'{col_nome} (Mediana)'] = np.nan
                linha[f'{col_nome} (DP)'] = np.nan
                
        dados_tabela.append(linha)
        print(f"✅ Processado: {nome_final} (Usou {textos_validos}/{total_textos} textos)")
        
    except Exception as e:
        print(f"❌ Erro ao processar {arquivo}: {e}")

# =============================================================================
# 4. FORMATAÇÃO E EXPORTAÇÃO
# =============================================================================
if dados_tabela:
    df_tabela = pd.DataFrame(dados_tabela)

    # Ordena a tabela pela Média do SBERT (do melhor para o pior)
    if 'SBERT (Média)' in df_tabela.columns:
        df_tabela = df_tabela.sort_values(by='SBERT (Média)', ascending=False)

    # Arredonda as colunas numéricas para 4 casas decimais
    cols_num = [c for c in df_tabela.columns if 'Média' in c or 'Mediana' in c or 'DP' in c]
    df_tabela[cols_num] = df_tabela[cols_num].map(lambda x: round(x, 4) if pd.notna(x) else np.nan)

    # Salva o arquivo CSV final (usando utf-8-sig para garantir leitura perfeita no Excel)
    nome_arquivo_saida = 'Tabela_Estatisticas_Modelos.csv'
    df_tabela.to_csv(nome_arquivo_saida, index=False, sep=';', decimal=',', encoding='utf-8-sig')

    print("\n" + "="*80)
    print(f"📊 SUCESSO! Arquivo '{nome_arquivo_saida}' gerado na sua pasta.")
    print("Métricas calculadas exclusivamente sobre as inferências íntegras (Higienizadas).")
    print("="*80)
else:
    print("⚠️ Nenhum dado válido foi processado.")