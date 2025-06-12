import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Assistente Médico Especializado",
    page_icon="🩺",
    layout="centered"
)

# --- TÍTULO E DESCRIÇÃO ---
st.title("🩺 Assistente Médico Especializado")
st.write(
    "Interface para interagir com o modelo de linguagem `mistral-7b-medico-finetuned`."
)
st.write(
    "Faça sua pergunta médica diretamente no campo abaixo."
)
st.write("---")

# --- 1. CARREGAMENTO DO MODELO (CACHEADO) ---
# Usamos o cache do Streamlit para carregar o modelo apenas uma vez.

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo fine-tuned e o tokenizador."""
    
    # Caminhos para os modelos
    modeloBase = "mistralai/Mistral-7B-Instruct-v0.2"
    modeloAdaptado = "mistral-7b-medico-finetuned"

    # Carrega o tokenizador
    tokenizador = AutoTokenizer.from_pretrained(modeloBase, trust_remote_code=True)
    tokenizador.pad_token = tokenizador.eos_token
    tokenizador.padding_side = "right"

    # Carrega o modelo base em 4-bit
    modelo_base = AutoModelForCausalLM.from_pretrained(
        modeloBase,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Combina o modelo base com os pesos do fine-tuning
    modelo_final = PeftModel.from_pretrained(modelo_base, modeloAdaptado)
    # Junta as camadas para otimizar a performance
    modelo_final = modelo_final.merge_and_unload()
    
    return modelo_final, tokenizador

# Exibe uma mensagem enquanto o modelo carrega
with st.spinner("Carregando o modelo especializado... Isso pode levar alguns minutos na primeira vez."):
    modelo, tokenizador = carregar_modelo()

st.success("✅ Modelo carregado com sucesso!")
st.write("---")

# --- 2. INTERFACE DO USUÁRIO ---

# Usando st.form para agrupar o campo e o botão
with st.form("chat_form"):
    st.subheader("Faça sua pergunta ao assistente")
    
    # Campo para a instrução (pergunta principal)
    instrucao_usuario = st.text_input(
        "Pergunta:", 
        placeholder="Ex: Quais são os sintomas de apendicite?"
    )
    
    # Botão de envio dentro do formulário
    submitted = st.form_submit_button("Gerar Resposta")

# --- 3. LÓGICA DE GERAÇÃO DE RESPOSTA ---

if submitted:
    if not instrucao_usuario:
        st.warning("Por favor, digite uma pergunta.")
    else:
        with st.spinner("O modelo está pensando..."):
            # Formata o prompt, agora sem o campo de contexto do usuário
            prompt = f"""### Instrução:
{instrucao_usuario}

### Contexto:


### Resposta:"""

            # Codifica o prompt e envia para a GPU
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

            # Exibe a resposta
            st.subheader("Resposta do Modelo:")
            st.markdown(resposta_limpa)
