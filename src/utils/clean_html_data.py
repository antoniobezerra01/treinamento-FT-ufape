import pandas as pd
import ftfy
from bs4 import BeautifulSoup

origem = ['data/in/2024_01_parecer_grupos.xlsx', 'data/in/2024_01_parecer_membros.xlsx', 'data/in/2024_03_parecer_grupos_v2.xlsx', 'data/in/2024_03_parecer_membros.xlsx']

print("Lendo o arquivo Excel...")

def limpeza_html(text):
    if pd.isna(text): return ""

    # 1. Remove HTML tags
    soup = BeautifulSoup(str(text), "html.parser")
    text_sem_tags = soup.get_text(separator=" ")

    # 2. O ftfy arruma acentos, entidades (cedilha) e mojibake (caracteres estranhos)
    return ftfy.fix_text(text_sem_tags)

def destino(origem_arquivo):
    return origem_arquivo.replace('data/in/', 'data/out/').replace('.xlsx', '_corrigido.csv')

for origem_arquivo in origem:
    df = pd.read_excel(origem_arquivo)
    print(f"Limpando o arquivo {origem_arquivo}...")
    if 'DS_JUSTIFICATIVA' in df.columns:
        df['DS_JUSTIFICATIVA'] = df['DS_JUSTIFICATIVA'].apply(limpeza_html)
    df.to_csv(destino(origem_arquivo), index=False)

print(f"Arquivos processados e salvos em data/out/.")