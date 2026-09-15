from unsloth import FastLanguageModel
import torch
import os

# --- CONFIGURAÇÃO ---
# Escolha qual versão converter:
# Opção A: O Checkpoint 40 (Recomendado para evitar overfitting/lobotomia)
modelo_entrada = "pnld_checkpoints/checkpoint-40"

# Opção B: O modelo final (Se o loss não tiver ficado muito baixo, tipo 0.05)
# modelo_entrada = "modelo_pnld_grupos2024_01_v1" 

# Nome do arquivo de saída (sem a extensão .gguf)
nome_saida = "PNLD_Juridico_Gemma2_9B"

# Quantização (q4_k_m é o padrão ouro para LM Studio: leve e inteligente)
modo_quantizacao = "q4_k_m" 

print(f"--- Carregando modelo de: {modelo_entrada} ---")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = modelo_entrada,
        max_seq_length = 8192,
        dtype = None,
        load_in_4bit = True,
    )
    
    print("Iniciando conversão para GGUF...")
    print("Isso pode demorar uns 10 a 15 minutos e usar bastante RAM.")

    # Esta função faz a fusão e a conversão automaticamente
    model.save_pretrained_gguf(
        nome_saida, 
        tokenizer, 
        quantization_method = modo_quantizacao
    )
    
    caminho_final = f"{nome_saida}/{nome_saida}-{modo_quantizacao}.gguf"
    print("-" * 40)
    print(f"SUCESSO! O arquivo foi criado em:")
    print(f"{caminho_final}")
    print("-" * 40)

except Exception as e:
    print(f"ERRO: {e}")