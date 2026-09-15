import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Lê o arquivo dependendo da extensão."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.jsonl':
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Formato {ext} não suportado.")

def merge_and_save(path1, path2, output_name="dataset_mesclado"):
    """
    Mescla dois arquivos e SALVA o resultado físico no disco.
    """
    df1 = load_data(path1)
    df2 = load_data(path2)
    
    df_merged = pd.concat([df1, df2], ignore_index=True)
    
    # COMANDO DE SALVAMENTO FÍSICO (Merge)
    df_merged.to_json(f"{output_name}.jsonl", orient='records', lines=True, force_ascii=False)
    print(f"✅ Arquivo mesclado salvo com sucesso: {output_name}.jsonl")
    
    return df_merged

def split_and_save(df, test_size=0.5, prefix="split_resultado"):
    """
    Divide o DataFrame e SALVA as duas partes como arquivos físicos no disco.
    """
    # Divide os dados (50/50 por padrão)
    df_a, df_b = train_test_split(df, test_size=test_size, random_state=42)
    
    # Nomes dos arquivos de saída
    file_a = f"{prefix}_treino.jsonl"
    file_b = f"{prefix}_teste.jsonl"
    
    # COMANDO DE SALVAMENTO FÍSICO (Split)
    df_a.to_json(file_a, orient='records', lines=True, force_ascii=False)
    df_b.to_json(file_b, orient='records', lines=True, force_ascii=False)
    
    print(f"✅ Divisão concluída!")
    print(f"   -> Arquivo 1 salvo: {file_a} ({len(df_a)} registros)")
    print(f"   -> Arquivo 2 salvo: {file_b} ({len(df_b)} registros)")
    
    return df_a, df_b

df_dataset2_full = pd.read_json('dataset_tcc_dataset2_reprovacoes_treino_460c2.jsonl', lines=True)
split_and_save(df_dataset2_full, prefix='dataset_tcc_dataset2_split')

df_dataset1_full = pd.read_json('dataset_tcc_dataset1_reprovacoes_teste_116c2.jsonl', lines=True)
merge_and_save('dataset_tcc_dataset1_reprovacoes_treino_116c1.jsonl', 'dataset_tcc_dataset2_split_treino.jsonl', 'dataset_tcc_dataset1_reprovacoes_treino_346c1')
