import os
import streamlit as st
from dotenv import load_dotenv
import time # Adicionado para o delay

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Título da Aplicação ---
st.title("🤖 Consulta de Dados (RAG) para LGPD")
st.write("Faça uma pergunta sobre os a LEI No 13.709, DE 14 DE AGOSTO DE 2018 disponível na Imprensa Nacional.")
st.write("---")

# --- Carregamento das Chaves de API ---
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openrouter_api_key = os.getenv("DEEPSEEK_API_KEY")

if not openai_api_key or not openrouter_api_key:
    st.error("⚠️ As chaves de API não foram encontradas. Por favor, configure seu arquivo .env.")
    st.stop()

# --- CONFIGURAÇÃO DOS MODELOS (LÓGICA HÍBRIDA) ---

# 1. Cliente para Embeddings (usando OpenAI diretamente)
# Conforme solicitado, este cliente se conecta diretamente à OpenAI.
embeddings_client = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=openai_api_key
)

# 2. Cliente para o LLM (usando OpenRouter)
# Este cliente se conecta ao OpenRouter para ter flexibilidade de modelos.
llm_client = ChatOpenAI(
    model="deepseek-chat",  # Alterado de "openai/gpt-3.5-turbo"
    api_key=openrouter_api_key,
    base_url="https://api.deepseek.com",
    temperature=0.3
)

# --- FUNÇÃO CACHEADA PARA CRIAR A PIPELINE RAG ---
# Usamos o cache do Streamlit para não reconstruir a base de dados a cada interação.
@st.cache_resource
def create_rag_pipeline():
    """
    Cria e retorna a pipeline RAG completa.
    Carrega docs, cria chunks, gera embeddings (OpenAI) e monta a cadeia (OpenRouter).
    """
    status_placeholder = st.empty() # Criar um placeholder para as mensagens de status

    document_path = "./documentos"
    if not os.path.exists(document_path) or not os.listdir(document_path):
        status_placeholder.warning("A pasta 'documentos' está vazia. Adicione arquivos PDF para começar.")
        return None

    status_placeholder.info("Carregando documentos...")
    # Carrega todos os PDFs da pasta
    all_docs = [PyPDFLoader(os.path.join(document_path, f)).load() for f in os.listdir(document_path) if f.endswith(".pdf")]
    # Concatena as listas de documentos
    docs = [item for sublist in all_docs for item in sublist]

    status_placeholder.info("Dividindo textos em trechos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    status_placeholder.info("Gerando embeddings (via OpenAI) e criando a base de conhecimento (FAISS)...")
    # Usa o cliente de embeddings da OpenAI
    vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings_client)

    status_placeholder.success("Base de conhecimento pronta!")
    time.sleep(2) # Pequeno delay para o usuário ver a mensagem de sucesso
    status_placeholder.empty() # Limpa as mensagens de status
    
    # Monta a cadeia de RetrievalQA usando o LLM do OpenRouter
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm_client,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return rag_chain

# --- INTERFACE DO USUÁRIO ---
rag_pipeline = create_rag_pipeline()

if rag_pipeline:
    with st.form(key="query_form"):
        user_question = st.text_input("Sua pergunta:")
        submit_button = st.form_submit_button("Buscar Resposta")

    if submit_button:
        if user_question:
            with st.spinner("Pensando..."):
                try:
                    response = rag_pipeline.invoke({"query": user_question})
                    st.subheader("Resposta:")
                    st.write(response.get("result", "Não foi possível obter uma resposta."))
                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
        else:
            st.warning("Por favor, digite uma pergunta.")
