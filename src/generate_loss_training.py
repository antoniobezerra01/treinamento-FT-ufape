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

# Estilização: Cores fixas para cada modelo
CORES_MODELOS = {
    "Llama-3.2-1B-Instruct-bnb-4bit": "#1f77b4",       # Azul
    "Llama-3.2-3B-Instruct-bnb-4bit": "#aec7e8",       # Azul Claro
    "Meta-Llama-3.1-8B-Instruct-bnb-4bit": "#2ca02c",  # Verde
    "gemma-3-1b-it-unsloth-bnb-4bit": "#d62728",       # Vermelho
    "gemma-3-4b-it-unsloth-bnb-4bit": "#ff9896",       # Vermelho Claro
    "gemma-3-12b-it-unsloth-bnb-4bit": "#9467bd",      # Roxo
    "gemma-2-9b-it-bnb-4bit": "#ff7f0e"                # Laranja
}

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

# 2. LOOP DE GERAÇÃO DOS GRÁFICOS
for cenario in CENARIOS:
    print(f"\n📊 Processando dados do Cenário: {cenario.upper()}")
    
    # Cria duas figuras separadas: Uma para Treino, outra para Validação
    fig_train, ax_train = plt.subplots(figsize=(12, 7))
    fig_eval, ax_eval = plt.subplots(figsize=(12, 7))
    
    modelos_plotados_train = 0
    modelos_plotados_eval = 0
    
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
            
            # Plota Training Loss se existir
            if e_treino and l_treino:
                ax_train.plot(e_treino, l_treino, label=label_limpo, color=CORES_MODELOS[modelo], linewidth=2.5, alpha=0.85)
                modelos_plotados_train += 1
                
            # Plota Validation Loss se existir
            if e_eval and l_eval:
                ax_eval.plot(e_eval, l_eval, label=label_limpo, color=CORES_MODELOS[modelo], linewidth=2.5, alpha=0.85)
                modelos_plotados_eval += 1
                
            print(f"  ✅ {label_limpo} processado com sucesso.")
        else:
            print(f"  ❌ Faltando pasta ou JSON para: {modelo}")
    
    # =============================================================================
    # 3. SALVAMENTO E ESTILIZAÇÃO DO GRÁFICO DE TREINO (TRAINING LOSS)
    # =============================================================================
    if modelos_plotados_train > 0:
        ax_train.set_title(f"Curva de Aprendizado (Training Loss) - Cenário {cenario.upper()}", fontsize=16, fontweight='bold', pad=15)
        ax_train.set_xlabel("Épocas de Treinamento", fontsize=14, labelpad=10)
        ax_train.set_ylabel("Perda de Treinamento (Train Loss)", fontsize=14, labelpad=10)
        ax_train.grid(True, linestyle='--', alpha=0.6)
        ax_train.tick_params(axis='both', which='major', labelsize=12)
        ax_train.legend(title="Modelos", title_fontsize='13', fontsize='11', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        
        fig_train.tight_layout()
        
        cam_train_png = os.path.join(BASE_DIR, f"Grafico_Training_Loss_{cenario.upper()}.png")
        cam_train_pdf = os.path.join(BASE_DIR, f"Grafico_Training_Loss_{cenario.upper()}.pdf")
        
        fig_train.savefig(cam_train_png, dpi=300, bbox_inches='tight')
        fig_train.savefig(cam_train_pdf, format='pdf', bbox_inches='tight')
        print(f"🎯 Gráfico de Treino salvo em: {cam_train_png}")
    
    plt.close(fig_train)

    # =============================================================================
    # 4. SALVAMENTO E ESTILIZAÇÃO DO GRÁFICO DE VALIDAÇÃO (EVALUATION LOSS)
    # =============================================================================
    if modelos_plotados_eval > 0:
        ax_eval.set_title(f"Curva de Validação (Evaluation Loss) - Cenário {cenario.upper()}", fontsize=16, fontweight='bold', pad=15)
        ax_eval.set_xlabel("Épocas de Treinamento", fontsize=14, labelpad=10)
        ax_eval.set_ylabel("Perda de Validação (Eval Loss)", fontsize=14, labelpad=10)
        ax_eval.grid(True, linestyle='--', alpha=0.6)
        ax_eval.tick_params(axis='both', which='major', labelsize=12)
        ax_eval.legend(title="Modelos", title_fontsize='13', fontsize='11', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        
        fig_eval.tight_layout()
        
        cam_eval_png = os.path.join(BASE_DIR, f"Grafico_Eval_Loss_{cenario.upper()}.png")
        cam_eval_pdf = os.path.join(BASE_DIR, f"Grafico_Eval_Loss_{cenario.upper()}.pdf")
        
        fig_eval.savefig(cam_eval_png, dpi=300, bbox_inches='tight')
        fig_eval.savefig(cam_eval_pdf, format='pdf', bbox_inches='tight')
        print(f"🎯 Gráfico de Validação salvo em: {cam_eval_png}")
        
    plt.close(fig_eval)

print("\n🚀 Script de Geração de Gráficos finalizado com sucesso!")