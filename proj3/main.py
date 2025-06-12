import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Carrega as variáveis de ambiente do arquivo .env
# Certifique-se de que seu OPENAI_API_KEY está neste arquivo.
load_dotenv()

# --- CONFIGURAÇÃO DO LLM ---
# Usaremos o modelo da OpenAI como o cérebro dos nossos agentes.
llm = ChatOpenAI(model="gpt-4.1-mini")

# --- DEFINIÇÃO DOS AGENTES ---

gerente_de_projetos = Agent(
  role='Gerente de Projetos de TI',
  goal='Interpretar o objetivo do usuário e criar um plano de execução passo a passo claro para um desenvolvedor.',
  backstory="""Você é um gerente de projetos experiente, especialista em traduzir necessidades de negócio em planos de ação técnicos. 
  Sua principal habilidade é criar um passo a passo detalhado para que os desenvolvedores possam executar as tarefas sem ambiguidades.""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)

desenvolvedor_tecnico = Agent(
  role='Desenvolvedor Python Sênior',
  goal='Escrever código Python completo e funcional com base em um plano fornecido.',
  backstory="""Você é um programador Python de elite. Você recebe um plano de ação e o transforma em código funcional, limpo e eficiente. 
  Você não faz perguntas, apenas executa o plano com perfeição.""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)

analista_de_resultados = Agent(
  role='Analista de Qualidade e Negócios',
  goal='Avaliar a saída do desenvolvedor, verificar se o objetivo original foi atingido e criar um relatório final claro e conciso.',
  backstory="""Você tem um olhar crítico e focado em resultados. Sua função é ser o "cliente final" da equipe, validando se a solução
  técnica resolve o problema de negócio. Você transforma dados brutos e códigos em relatórios que qualquer pessoa possa entender.""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)

# --- DEFINIÇÃO DAS TAREFAS ---
tarefa_de_planejamento = Task(
  description=(
    "1. Analise o seguinte objetivo do usuário: '{objetivo_usuario}'.\n"
    "2. Crie um plano de execução passo a passo, com tarefas específicas e claras para o Desenvolvedor Técnico.\n"
    "3. Defina qual será o resultado final esperado (ex: um script Python, uma análise de texto, um arquivo CSV)."
  ),
  expected_output='Um documento de texto detalhado contendo um plano de ação claro e o formato do resultado final esperado.',
  agent=gerente_de_projetos
)

tarefa_de_execucao = Task(
  description=(
    "Com base no plano recebido do Gerente de Projetos, execute a tarefa técnica. \n"
    "Escreva todo o código Python necessário para atingir o objetivo. O código deve ser completo e funcional."
  ),
  expected_output='O resultado técnico da tarefa, que deve ser um código-fonte completo e funcional em Python.',
  agent=desenvolvedor_tecnico,
  context=[tarefa_de_planejamento]
)

tarefa_de_analise = Task(
  description=(
    "1. Revise o trabalho executado pelo Desenvolvedor Técnico, com base no plano original do Gerente de Projetos.\n"
    "2. Verifique se o objetivo inicial do usuário foi completamente atingido e se o código está funcional.\n"
    "3. Escreva um relatório final resumindo o que foi feito, os resultados obtidos e como executar o código, se aplicável."
  ),
  expected_output='Um relatório final em formato de texto, resumindo todo o processo e os resultados.',
  agent=analista_de_resultados,
  context=[tarefa_de_execucao]
)

# --- MONTAGEM DA EQUIPE (CREW) ---
equipe = Crew(
  agents=[gerente_de_projetos, desenvolvedor_tecnico, analista_de_resultados],
  tasks=[tarefa_de_planejamento, tarefa_de_execucao, tarefa_de_analise],
  process=Process.sequential,
  verbose=True
)

# --- EXECUÇÃO ---
objetivo_do_usuario = "Crie um jogo da forca simples em Python que possa ser jogado no terminal. O jogo deve ter uma lista interna de palavras, permitir que o jogador adivinhe as letras e indicar se o jogador ganhou ou perdeu."

print("Equipe pronta! Iniciando a execução da tarefa...")
print(f"Objetivo: {objetivo_do_usuario}")
print("--------------------------------------------------")

resultado = equipe.kickoff(inputs={'objetivo_usuario': objetivo_do_usuario})

print("\n\n--------------------------------------------------")
print("Execução concluída!")
print("Resultado Final:")
print(resultado)
