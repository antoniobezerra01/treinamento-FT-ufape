from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import gc

# --- 1. LISTA DE MODELOS  ---
lista_modelos = [
    # 🦙 Família Meta (Llama 3.1 e 3.2)
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",

    # 💎 Família Google (Gemma 3 e Gemma 2)
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit"
]

# --- ARQUIVO DE DATASET ---
arquivo_treino = "dataset_tcc_dataset2_reprovacoes_treino_460c2.jsonl"

# Carrega o dataset completo de treino
dataset_bruto = load_dataset("json", data_files=arquivo_treino, split="train")

# Faz a divisão (Split): 80% Treino e 20% Validação
dataset_dividido = dataset_bruto.train_test_split(test_size=0.2, seed=42)

dataset_treino_bruto = dataset_dividido["train"]
dataset_validacao_bruto = dataset_dividido["test"]

print(f"📊 Divisão Concluída: {len(dataset_treino_bruto)} exemplos para Treino, {len(dataset_validacao_bruto)} para Validação.")

prompt_tcc = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# --- 2. O LOOP DE TREINAMENTO ---
for modelo_base in lista_modelos:
    # Cria o nome da pasta dinamicamente
    nome_curto = modelo_base.split("/")[-1]
    pasta_saida = f"modelo_pnld_{nome_curto}_c2_validation"
    
    print("\n" + "="*80)
    print(f"🚀 Iniciando Fine-Tuning do TCC com o modelo: {modelo_base}")
    print("="*80)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = modelo_base,
            max_seq_length = 8192, 
            dtype = None,
            load_in_4bit = True,
        )

        # Configuração do PEFT/LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8, 
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0.05,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )

        EOS_TOKEN = tokenizer.eos_token
        def formatar_prompt(exemplos):
            instrucoes = exemplos["instruction"]
            inputs     = exemplos["input"]
            outputs    = exemplos["output"]
            textos = []
            for inst, inp, out in zip(instrucoes, inputs, outputs):
                texto = prompt_tcc.format(inst, inp, out) + EOS_TOKEN
                textos.append(texto)
            return { "text" : textos, }

        # Formata ambos os datasets treino reduzido e o de validação
        dataset_treino_formatado = dataset_treino_bruto.map(formatar_prompt, batched=True)
        dataset_validacao_formatado  = dataset_validacao_bruto.map(formatar_prompt, batched=True)

        # --- HIPERPARÂMETROS ---
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset_treino_formatado,
            eval_dataset = dataset_validacao_formatado, # <-- Usa a fatia de 20% para validação
            dataset_text_field = "text",
            max_seq_length = 8192,
            packing = False, 
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                per_device_eval_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 10,
                num_train_epochs = 4,
                learning_rate = 2e-5,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                
                # --- CONFIGURAÇÕES DE AVALIAÇÃO ---
                eval_strategy = "steps", # Avalia a cada X passos
                eval_steps = 10,         # A cada 10 passos, calcula a Validation Loss
                save_strategy = "steps", # Salva checkpoints junto com a avaliação
                save_steps = 10,         # Salva a cada 10 passos
                # ----------------------------------
                
                optim = "adamw_8bit",
                weight_decay = 0.01,
                output_dir = pasta_saida,
            ),
        )

        print(f"⏳ Iniciando o treinamento de {nome_curto} (Aguarde...).")
        resultado_treino = trainer.train()

        # --- SALVAR O MODELO ---
        print(f"✅ Treinamento concluído! Salvando modelo em: {pasta_saida}")
        model.save_pretrained(pasta_saida)
        tokenizer.save_pretrained(pasta_saida)

    except Exception as e:
        print(f"❌ ERRO FATAL ao processar o modelo {modelo_base}: {e}")

    finally:
        # --- LIMPEZA DE MEMÓRIA ---
        print("🧹 Limpando memória da GPU para o próximo modelo...")
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        if 'trainer' in locals(): del trainer
        gc.collect()             
        torch.cuda.empty_cache() 
        print("✨ Memória liberada com sucesso!\n")

print("🎉 TODOS OS MODELOS FORAM TREINADOS COM SUCESSO!")