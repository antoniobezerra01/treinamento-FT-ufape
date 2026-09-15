import pandas as pd
import ftfy
from bs4 import BeautifulSoup
from pathlib import Path

# 1. Mapeamento dinâmico dos diretórios
DIR_IN = Path('data/in')
DIR_OUT = Path('data/out/v2')

# Garante que a pasta de saída exista (cria se não existir)
DIR_OUT.mkdir(parents=True, exist_ok=True)

def limpeza_html(text):
    """Remove tags HTML e corrige problemas de encoding (mojibake)."""
    if pd.isna(text): 
        return ""

    # Remove HTML tags
    soup = BeautifulSoup(str(text), "html.parser")
    text_sem_tags = soup.get_text(separator=" ")

    # O ftfy arruma acentos, entidades e caracteres corrompidos
    return ftfy.fix_text(text_sem_tags)

# 2. Busca todos os arquivos .xlsx dinamicamente dentro da pasta
arquivos_origem = list(DIR_IN.glob('*.xlsx'))

print(f"🚀 Iniciando pipeline de limpeza. Encontrados {len(arquivos_origem)} arquivos em {DIR_IN}/\n")

if not arquivos_origem:
    print("⚠️ Nenhum arquivo .xlsx foi encontrado na pasta de origem.")
else:
    for arquivo in arquivos_origem:
        print(f"🔄 Lendo e limpando: {arquivo.name} ...")
        
        # Lê o Excel
        df = pd.read_excel(arquivo)
        
        # Aplica a limpeza se a coluna existir
        if 'DS_JUSTIFICATIVA' in df.columns:
            df['DS_JUSTIFICATIVA'] = df['DS_JUSTIFICATIVA'].apply(limpeza_html)
        
        # 3. Gera o nome de saída dinamicamente
        # .stem pega apenas o nome do arquivo sem a extensão (ex: '2024_01_parecer_grupos')
        nome_novo = f"{arquivo.stem}_corrigido.csv"
        caminho_destino = DIR_OUT / nome_novo
        
        # Salva o resultado
        df.to_csv(caminho_destino, index=False, encoding='utf-8')
        print(f"✅ Salvo com sucesso em: {caminho_destino}")

print("\n🎉 Todos os arquivos foram processados e salvos!")