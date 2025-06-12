import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. CARREGAR O MODELO E O TOKENIZADOR ---

# Modelo base original
modeloBase = "mistralai/Mistral-7B-Instruct-v0.2"
# Pasta onde seu modelo adaptado foi salvo
modeloAdaptado = "mistral-7b-medico-finetuned"

# Carrega o tokenizador
print("Carregando tokenizador...")
tokenizador = AutoTokenizer.from_pretrained(modeloBase, trust_remote_code=True)
tokenizador.pad_token = tokenizador.eos_token
tokenizador.padding_side = "right"

# Carrega o modelo base em 4-bit
print("Carregando modelo base... Isso pode demorar um pouco.")
modelo = AutoModelForCausalLM.from_pretrained(
    modeloBase,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Combina o modelo base com os pesos adaptados do seu fine-tuning
print("Aplicando adaptações do fine-tuning...")
modelo = PeftModel.from_pretrained(modelo, modeloAdaptado)
# Junta as camadas para otimizar a performance
modelo = modelo.merge_and_unload()


# --- 2. FUNÇÃO PARA GERAR RESPOSTAS (SIMPLIFICADA) ---

def gerar_resposta(instrucao):
    """Formata o prompt (sem contexto) e gera uma resposta do modelo."""
    # Mantemos a estrutura do prompt, pois o modelo foi treinado com ela,
    # mas o campo de contexto fica vazio.
    prompt = f"""### Instrução:
{instrucao}

### Contexto:


### Resposta:"""

    # Codifica o prompt
    inputs = tokenizador(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

    # Gera a resposta
    outputs = modelo.generate(
        **inputs, 
        max_new_tokens=512,
        pad_token_id=tokenizador.eos_token_id
    )
    
    # Decodifica a resposta, pulando o prompt inicial
    texto_gerado = tokenizador.decode(outputs[0], skip_special_tokens=True)
    inicio_resposta = texto_gerado.find("### Resposta:") + len("### Resposta:")
    resposta_limpa = texto_gerado[inicio_resposta:].strip()

    # Imprime a resposta formatada
    print("\n--- RESPOSTA DO MODELO ---")
    print(resposta_limpa)
    print("--------------------------\n")


# --- 3. LOOP INTERATIVO PARA TESTES (SIMPLIFICADO) ---

print("Modelo médico carregado. Faça suas perguntas.")
print("Digite 'sair' para terminar.")
print("-" * 30)

while True:
    # Apenas uma entrada do usuário é necessária agora
    instrucao_usuario = input("Pergunta: ")
    if instrucao_usuario.lower() == 'sair':
        break
    
    gerar_resposta(instrucao_usuario)