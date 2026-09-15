import pandas as pd
import re
import glob
import os
import collections

# Margem aceitável de itens a mais que a IA pode gerar em relação ao humano
MARGEM_ALUCINACAO = 10 

def contar_itens_edital(texto):
    texto = str(texto)
    return len(re.findall(r'\d+\.\d+\.\d+', texto))

def detectar_loop_frasal(texto):
    """
    Retorna True se o modelo repetir o mesmo esqueleto de frase 4 vezes ou mais,
    ignorando mudanças apenas nos números (Alucinação Parametrizada).
    """
    texto = str(texto)
    frases_cruas = re.split(r'\.\s+', texto)
    
    frases_limpas = []
    for f in frases_cruas:
        # Substitui todos os dígitos por 'X' para capturar o "molde" da frase
        f_esqueleto = re.sub(r'\d+', 'X', f).strip()
        
        # Só analisa frases que tenham mais de 15 caracteres
        if len(f_esqueleto) > 15:
            frases_limpas.append(f_esqueleto)
            
    if len(frases_limpas) < 5:
        return False
        
    contagem_frases = collections.Counter(frases_limpas)
    for frase, qtd in contagem_frases.items():
        if qtd >= 4:
            return True
            
    return False

def detectar_vazamento_prompt(texto):
    """Retorna True se o modelo vazou as marcações do prompt no texto gerado."""
    texto = str(texto)
    termos_proibidos = ["### Instruction:", "### Response:", "### Input:", "Você é um Revisor Técnico"]
    return any(termo in texto for termo in termos_proibidos)

print("🔍 Iniciando Auditoria Tripla (Numérica, Lexical e Vazamento de Prompt) e Separação de Base Limpa...\n")

arquivos_csv = glob.glob("**/*Resultados_*.csv", recursive=True)

if not arquivos_csv:
    todos_csvs = glob.glob("**/*.csv", recursive=True)
    arquivos_csv = [f for f in todos_csvs if "Resultados" in os.path.basename(f)]

if not arquivos_csv:
    print("⚠️ Nenhum arquivo de resultado encontrado.")
else:
    resultados_finais = []
    lista_casos_problematicos = []
    lista_casos_validos = [] 

    # Armazenamento de logs
    log_terminal_latex = [] 
    
    print(f"📂 Encontrados {len(arquivos_csv)} arquivos para análise. Processando...\n")

    for arquivo in arquivos_csv:
        try:
            df = pd.read_csv(arquivo)
            
            if 'Rascunho_Original' not in df.columns or 'Parecer_Gerado_IA' not in df.columns:
                continue
                
            total_textos = len(df)
            
            # 1. Aplica as 3 validações de auditoria
            df['Qtd_Humano'] = df['Rascunho_Original'].apply(contar_itens_edital)
            df['Qtd_IA'] = df['Parecer_Gerado_IA'].apply(contar_itens_edital)
            
            df['Loop_Numerico'] = df['Qtd_IA'] > (df['Qtd_Humano'] + MARGEM_ALUCINACAO)
            df['Loop_Lexical'] = df['Parecer_Gerado_IA'].apply(detectar_loop_frasal)
            df['Vazamento'] = df['Parecer_Gerado_IA'].apply(detectar_vazamento_prompt)
            
            # O Veredito final
            df['Alucinacao_Severa'] = df['Loop_Numerico'] | df['Loop_Lexical'] | df['Vazamento']
            
            falhas_reais_ia = df['Alucinacao_Severa'].sum()
            perc_falhas = (falhas_reais_ia / total_textos) * 100
            
            # Degeneração Repetitiva = Ocorreu Loop Numérico OU Loop Lexical
            qtd_deg_rep = (df['Loop_Numerico'] | df['Loop_Lexical']).sum()
            qtd_vazamento = df['Vazamento'].sum()
            
            # 2. Formatação do Nome e Tags
            nome_limpo = os.path.basename(arquivo).replace(".csv", "")
            cenario = "??"
            if nome_limpo.lower().endswith(("_c1", "-c1")):
                cenario = "C1"
                nome_limpo = nome_limpo[:-3]
            elif nome_limpo.lower().endswith(("_c2", "-c2")):
                cenario = "C2"
                nome_limpo = nome_limpo[:-3]
                
            if "modelo_pnld_" in nome_limpo:
                nome_limpo = nome_limpo.replace("Resultados_modelo_pnld_", "[AJUSTADO] ")
            elif "Resultados_Base_" in nome_limpo:
                nome_limpo = nome_limpo.replace("Resultados_Base_", "")
                match_shot = re.search(r'(\d+)[-_]?shot', nome_limpo, re.IGNORECASE)
                if match_shot:
                    num_shots = match_shot.group(1)
                    nome_sem_shot = re.sub(r'[-_]?\d+[-_]?shot', '', nome_limpo, flags=re.IGNORECASE).strip("_ ")
                    nome_limpo = f"[{num_shots}-SHOT]   {nome_sem_shot}"
                else:
                    nome_limpo = f"[0-SHOT]   {nome_limpo}"
            
            nome_final = f"[{cenario}] {nome_limpo}"
            
            # Salva no log isolado para não alterar a estrutura original
            log_terminal_latex.append({
                "Configuracao": nome_final,
                "Degeneracao": qtd_deg_rep,
                "Vazamento": qtd_vazamento
            })
            
            resultados_finais.append({
                "Configuracao": nome_final,
                "Total Textos": total_textos,
                "Falhas de Geracao": f"{falhas_reais_ia} ({perc_falhas:.1f}%)",
                "taxa_ordenacao": perc_falhas 
            })
            
            # 3. Separa os Problemáticos
            df_erros = df[df['Alucinacao_Severa'] == True].copy()
            if not df_erros.empty:
                df_erros.insert(0, 'Modelo_Configuracao', nome_final)
                
                def definir_tipo_falha(row):
                    falhas = []
                    if row['Loop_Numerico']: falhas.append("Loop Numérico")
                    if row['Loop_Lexical']: falhas.append("Loop Lexical (Frases)")
                    if row['Vazamento']: falhas.append("Vazamento de Prompt")
                    return " + ".join(falhas)

                df_erros['Tipo_de_Falha'] = df_erros.apply(definir_tipo_falha, axis=1)
                lista_casos_problematicos.append(df_erros)
                
            # 4. Separa os Válidos (Base Limpa)
            df_validos = df[df['Alucinacao_Severa'] == False].copy()
            if not df_validos.empty:
                df_validos.insert(0, 'Modelo_Configuracao', nome_final)
                lista_casos_validos.append(df_validos)
            
        except Exception as e:
            print(f"❌ Erro ao processar {arquivo}: {e}")

    # =========================================================
    # EXPORTAÇÃO DOS 3 ARQUIVOS CSV
    # =========================================================
    if resultados_finais:
        # Exporta 1: Resumo
        df_resumo = pd.DataFrame(resultados_finais)
        df_resumo = df_resumo.sort_values(by="taxa_ordenacao", ascending=False).drop(columns=["taxa_ordenacao"])
        
        print("-" * 105)
        print(df_resumo.to_string(index=False))
        print("-" * 105)
        
        df_resumo.to_csv("Resumo_Auditoria_Alucinacoes_v2.csv", index=False, encoding='utf-8-sig')
        print("\n✅ [1/3] Arquivo salvo: Resumo_Auditoria_Alucinacoes_v2.csv")

        # Exporta 2: Casos Problemáticos
        if lista_casos_problematicos:
            df_todos_erros = pd.concat(lista_casos_problematicos, ignore_index=True)
            colunas_foco = ['Modelo_Configuracao', 'Tipo_de_Falha', 'Qtd_Humano', 'Qtd_IA', 'Rascunho_Original', 'Parecer_Gerado_IA']
            outras_colunas = [c for c in df_todos_erros.columns if c not in colunas_foco and c not in ['Loop_Lexical', 'Loop_Numerico', 'Vazamento', 'Alucinacao_Severa']]
            df_todos_erros = df_todos_erros[colunas_foco + outras_colunas]
            
            df_todos_erros.to_csv("Relatorio_Casos_Problematicos_v2.csv", index=False, encoding='utf-8-sig')
            print(f"🚨 [2/3] Arquivo salvo: Relatorio_Casos_Problematicos_v2.csv ({len(df_todos_erros)} textos reprovados)")
        else:
            print("\n🎉 [2/3] Nenhum caso problemático detectado!")

        if lista_casos_validos:
            df_todos_validos = pd.concat(lista_casos_validos, ignore_index=True)
            colunas_foco_validos = ['Modelo_Configuracao', 'Qtd_Humano', 'Qtd_IA', 'Rascunho_Original', 'Parecer_Gerado_IA']
            outras_colunas_validos = [c for c in df_todos_validos.columns if c not in colunas_foco_validos and c not in ['Loop_Lexical', 'Loop_Numerico', 'Vazamento', 'Alucinacao_Severa']]
            df_todos_validos = df_todos_validos[colunas_foco_validos + outras_colunas_validos]
            
            df_todos_validos.to_csv("Relatorio_Casos_Validos_v2.csv", index=False, encoding='utf-8-sig')
            print(f"🌟 [3/3] Arquivo salvo: Relatorio_Casos_Validos_v2.csv")

    print("\n" + "═"*80)
    print("📋 Contagem analise de alucinações.")
    print("═"*80)
    # Ordena para facilitar a busca alfabética no terminal
    for item in sorted(log_terminal_latex, key=lambda x: x['Configuracao']):
        print(f"Modelo/Configuração: {item['Configuracao']}")
        print(f"   -> Degeneração Repetitiva : {item['Degeneracao']}")
        print(f"   -> Vazamento de Prompt    : {item['Vazamento']}")
        print("-" * 50)