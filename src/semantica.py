import pandas as pd
import glob
import spacy
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ==========================================
# 1. Carregando os Modelos Semânticos
# ==========================================
print("⏳ Carregando modelo SpaCy (pt_core_news_lg)...")
nlp = spacy.load("pt_core_news_lg")

print("⏳ Carregando modelo SBERT (multilíngue)...")
# Usamos um modelo multilíngue leve e poderoso muito usado no Brasil
sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Procura todos os CSVs de resultados na pasta
arquivos = glob.glob("Resultados_*c*.csv")

# ==========================================
# 2. Processamento dos CSVs
# ==========================================
for arquivo in arquivos:
    print("\n" + "="*60)
    print(f"🚀 Processando arquivo: {arquivo}")
    
    # Lê o CSV ignorando linhas mal formatadas, se houver
    df = pd.read_csv(arquivo, on_bad_lines='skip')
    
    # Cria as colunas caso ainda não existam no CSV
    if 'Similaridade_SpaCy' not in df.columns:
        df['Similaridade_SpaCy'] = 0.0
    if 'Similaridade_SBERT' not in df.columns:
        df['Similaridade_SBERT'] = 0.0
        
    df['Similaridade_SpaCy'] = df['Similaridade_SpaCy'].astype(float)
    df['Similaridade_SBERT'] = df['Similaridade_SBERT'].astype(float)

    # Itera sobre cada linha com uma barra de progresso
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculando"):
        
        # Pula a linha se já tiver calculado antes (Sistema Anti-Crash)
        if df.at[index, 'Similaridade_SBERT'] > 0:
            continue
            
        texto_gerado = str(row['Parecer_Gerado_IA'])
        texto_real = str(row['Parecer_Real_GroundTruth'])
        
        
        # Métrica 1: SpaCy (Word Embeddings Clássico)
        
        doc_gerado = nlp(texto_gerado)
        doc_real = nlp(texto_real)
        
        # Verifica se as palavras existem no vocabulário para evitar divisão por zero
        if doc_gerado.vector_norm and doc_real.vector_norm:
            sim_spacy = doc_gerado.similarity(doc_real)
        else:
            sim_spacy = 0.0
            
        
        # Métrica 2: SBERT (Sentence Embeddings Profundo)
        
        # Codifica as duas frases em tensores e calcula o cosseno entre elas
        emb_gerado = sbert.encode(texto_gerado, convert_to_tensor=True)
        emb_real = sbert.encode(texto_real, convert_to_tensor=True)
        
        # util.cos_sim retorna uma matriz 1x1, usamos .item() para pegar o número float
        sim_sbert = util.cos_sim(emb_gerado, emb_real).item()
        
        # Atualiza os valores na tabela
        df.at[index, 'Similaridade_SpaCy'] = sim_spacy
        df.at[index, 'Similaridade_SBERT'] = sim_sbert
        
    # Salva o CSV sobrescrevendo o arquivo com as novas colunas
    df.to_csv(arquivo, index=False, encoding='utf-8')
    print(f"✅ Salvo e Atualizado com Sucesso: {arquivo}")

print("\n🎉 Todas as métricas semânticas profundas foram adicionadas aos CSVs!")