import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- OTIMIZAÇÕES DE MEMÓRIA ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()  # Limpa cache da GPU
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)  # Reduzido para 85% da memória da GPU

# --- 1. CONFIGURAÇÕES INICIAIS ---
modeloBase = "mistralai/Mistral-7B-Instruct-v0.2"
caminhoDoDataset = "./dataset/medDataset_processed.csv"
nomeNovoModelo = "mistral-7b-medico-finetuned"

# --- 2. PREPARAÇÃO DO DATASET ---
dataset = load_dataset('csv', data_files=caminhoDoDataset, split='train')
# Usar apenas uma pequena parte do dataset para teste
dataset = dataset.select(range(min(1000, len(dataset))))  # Usa apenas 1000 exemplos para teste

def formatarPrompt(exemplo):
    return f"""### Tipo de Pergunta:
{exemplo['qtype']}

### Pergunta:
{exemplo['Question']}

### Resposta:
{exemplo['Answer']}"""

# --- 3. CONFIGURAÇÃO DO MODELO E TOKENIZER ---
configuracaoBnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Mudado para True para economizar mais memória
)
modelo = AutoModelForCausalLM.from_pretrained(
    modeloBase,
    quantization_config=configuracaoBnb,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Força o uso de bfloat16
    low_cpu_mem_usage=True,  # Otimiza uso de memória da CPU
    trust_remote_code=True,
)
modelo.config.use_cache = False
modelo.config.pretraining_tp = 1

# Preparar o modelo para treinamento ANTES de aplicar LoRA
modelo.train()
for param in modelo.parameters():
    param.requires_grad = False  # Congela o modelo base
tokenizador = AutoTokenizer.from_pretrained(modeloBase, trust_remote_code=True)
tokenizador.pad_token = tokenizador.eos_token
tokenizador.padding_side = "right"

# --- 4. CONFIGURAÇÃO DO PEFT / LoRA ---
configuracaoLora = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,  # Reduzido ainda mais para 8 para economizar memória
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # Reduzido para apenas q_proj e v_proj
)
modelo = get_peft_model(modelo, configuracaoLora)

# Forçar habilitação de gradientes para os parâmetros LoRA
modelo.enable_adapter_layers()
modelo.train()

# Garantir que os parâmetros LoRA são treináveis
modelo.print_trainable_parameters()  # Para debug

# Verificar se algum parâmetro está sendo treinado
trainable_params = []
for name, param in modelo.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)

if not trainable_params:
    print("ERRO: Nenhum parâmetro está marcado como treinável!")
else:
    print(f"Total de parâmetros treináveis encontrados: {len(trainable_params)}")
    print("Configuração LoRA aplicada com sucesso!")

# --- 5. CONFIGURAÇÃO DO TREINAMENTO ---
argumentosDeTreinamento = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Mantém em 1 para RTX 3060
    gradient_accumulation_steps=16,  # Aumentado para 16 para compensar o batch menor
    optim="paged_adamw_8bit",  # Mudado para 8bit para economizar mais memória
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=100,  # Reduzido ainda mais para teste
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    dataloader_pin_memory=False,  # Reduz uso de memória
    remove_unused_columns=False,  # Previne problemas com colunas
    logging_steps=5,
    save_steps=50,
    eval_steps=50,
)

# --- 6. CRIAÇÃO DO TRAINER ---
treinador = SFTTrainer(
    model=modelo,
    train_dataset=dataset,
    peft_config=configuracaoLora,
    formatting_func=formatarPrompt,
    args=argumentosDeTreinamento,
)

# --- 7. EXECUÇÃO DO TREINAMENTO ---
print("🚀 INICIANDO O FINE-TUNING...")
treinador.train()
print("✅ FINE-TUNING CONCLUÍDO!")

# --- 8. SALVAR O MODELO ---
print("💾 SALVANDO O MODELO ADAPTADO...")
treinador.model.save_pretrained(nomeNovoModelo)
tokenizador.save_pretrained(nomeNovoModelo)
print(f"Modelo salvo em '{nomeNovoModelo}'")