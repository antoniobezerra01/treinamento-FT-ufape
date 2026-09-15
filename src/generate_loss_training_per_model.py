import os
import json
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURAÇÕES DOS DADOS
# =============================================================================
# Caminho base do seu projeto
BASE_DIR = r"C:\Users\Antonio Bezerra\OneDrive\Documentos\TCC-Alvaro-Fine-tunning\Code\modelos_ajustados"

CENARIOS = ["c1", "c2"]

MODELOS = [
    "Llama-3.2-1B-Instruct-bnb-4bit",
    "Llama-3.2-3B-Instruct-bnb-4bit",
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "gemma-3-1b-it-unsloth-bnb-4bit",
    "gemma-3-4b-it-unsloth-bnb-4bit",
    "gemma-3-12b-it-unsloth-bnb-4bit",
    "gemma-2-9b-it-bnb-4bit"
]

def extrair_dados_losses(caminho_json):
    """Lê o trainer_state.json e retorna listas separadas para Treino e Validação."""
    try:
        with open(caminho_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            
        epocas_treino = []
        losses_treino = []
        
        epocas_eval = []
        losses_eval = []
        
        for entrada in dados.get("log_history", []):
            # Extrai Training Loss
            if "loss" in entrada and "epoch" in entrada:
                epocas_treino.append(entrada["epoch"])
                losses_treino.append(entrada["loss"])
                
            # Extrai Validation Loss
            if "eval_loss" in entrada and "epoch" in entrada:
                epocas_eval.append(entrada["epoch"])
                losses_eval.append(entrada["eval_loss"])
                
        return epocas_treino, losses_treino, epocas_eval, losses_eval
    
    except Exception as e:
        print(f"Erro ao ler {caminho_json}: {e}")
        return None, None, None, None

# =============================================================================
# 2. LOOP DE GERAÇÃO DOS GRÁFICOS (UM POR MODELO E CENÁRIO)
# =============================================================================
for cenario in CENARIOS:
    print(f"\n📊 Processando dados do Cenário: {cenario.upper()}")
    
    for modelo in MODELOS:
        nome_pasta_base = f"modelo_pnld_{modelo}_{cenario}_validation"
        pasta_modelo = os.path.join(BASE_DIR, nome_pasta_base)
        caminho_arquivo = os.path.join(pasta_modelo, "trainer_state.json")
        
        # Lógica inteligente para buscar o checkpoint mais atualizado
        if not os.path.exists(caminho_arquivo) and os.path.exists(pasta_modelo):
            pastas_checkpoints = [d for d in os.listdir(pasta_modelo) if d.startswith("checkpoint-")]
            if pastas_checkpoints:
                pastas_checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
                caminho_arquivo = os.path.join(pasta_modelo, pastas_checkpoints[0], "trainer_state.json")
        
        if os.path.exists(caminho_arquivo):
            e_treino, l_treino, e_eval, l_eval = extrair_dados_losses(caminho_arquivo)
            label_limpo = modelo.split("-bnb")[0].replace("-it-unsloth", "").replace("-Instruct", "")
            
            # Só gera o gráfico se existirem dados (mesmo que seja só treino)
            if (e_treino and l_treino) or (e_eval and l_eval):
                # Cria a figura para o modelo atual
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plota Training Loss (Cor primária: Azul)
                if e_treino and l_treino:
                    ax.plot(e_treino, l_treino, label="Training Loss", color="#1f77b4", linewidth=2.5, alpha=0.85)
                    
                # Plota Validation Loss (Cor secundária: Laranja/Vermelho para dar contraste)
                if e_eval and l_eval:
                    ax.plot(e_eval, l_eval, label="Validation Loss", color="#ff7f0e", linewidth=2.5, alpha=0.85, linestyle='--')
                
                # Estilização
                ax.set_title(f"Convergência de Treino vs Validação\n{label_limpo} ({cenario.upper()})", fontsize=16, fontweight='bold', pad=15)
                ax.set_xlabel("Épocas de Treinamento", fontsize=14, labelpad=10)
                ax.set_ylabel("Perda (Loss)", fontsize=14, labelpad=10)
                
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.legend(fontsize='12', loc='upper right')
                
                fig.tight_layout()
                
                # Salva o arquivo com o nome específico do modelo
                nome_base_arquivo = f"Grafico_Loss_{label_limpo}_{cenario.upper()}"
                cam_png = os.path.join(BASE_DIR, f"{nome_base_arquivo}.png")
                cam_pdf = os.path.join(BASE_DIR, f"{nome_base_arquivo}.pdf")
                
                fig.savefig(cam_png, dpi=300, bbox_inches='tight')
                fig.savefig(cam_pdf, format='pdf', bbox_inches='tight')
                
                plt.close(fig)
                print(f"  ✅ Gráfico gerado para: {label_limpo}")
            else:
                print(f"  ⚠️ Dados insuficientes para plotar: {modelo}")
        else:
            print(f"  ❌ Faltando pasta ou JSON para: {modelo}")

print("\n🚀 Script de Geração de Gráficos Individuais finalizado com sucesso!")