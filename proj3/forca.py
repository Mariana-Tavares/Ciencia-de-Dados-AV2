"""
forca.py - Jogo da Forca em Python                                                                                                                

Como executar:
No terminal, execute o comando:
    python forca.py

Descrição:
Este script implementa um jogo da forca simples que funciona no terminal. O jogador deve tentar adivinhar
uma palavra sorteada aleatoriamente através da digitação de letras. O jogo informa o estado atual da palavra,
o número de tentativas restantes, e imprime mensagens de vitória ou derrota. Ao término, o jogador pode optar
por jogar novamente.

Requisitos:
- Python 3.6 ou superior.
- Não depende de bibliotecas externas além da biblioteca padrão.
"""

import random
import string

# Lista fixa de palavras para o jogo
PALAVRAS = [
    "python",
    "programa",
    "desenvolvimento",
    "terminal",
    "jogo",
    "computador",
    "software",
    "usuario",
    "teclado",
    "monitor"
]

def escolher_palavra(palavras):
    """Seleciona uma palavra aleatória da lista fornecida."""
    return random.choice(palavras).lower()

def inicializar_espaco_oculto(palavra):
    """Inicializa o espaço oculto da palavra usando underscores para letras não reveladas."""
    return ["_" if letra.isalpha() else letra for letra in palavra]

def mostrar_estado(palavra_oculta, tentativas_restantes, letras_tentadas):
    """Exibe o estado atual do jogo para o jogador."""
    print("\nPalavra: " + " ".join(palavra_oculta))
    print(f"Tentativas restantes: {tentativas_restantes}")
    print(f"Letras tentadas: {', '.join(sorted(letras_tentadas)) if letras_tentadas else 'Nenhuma'}")

def obter_letra_valida(letras_tentadas):
    """
    Solicita ao jogador que digite uma letra válida.
    Aceita apenas uma única letra que ainda não tenha sido tentada.
    Ignora case.
    """
    while True:
        entrada = input("Digite uma letra: ").strip().lower()
        if len(entrada) != 1:
            print("Por favor, digite apenas uma única letra.")
            continue
        if entrada not in string.ascii_lowercase:
            print("Entrada inválida. Digite uma letra do alfabeto (a-z).")
            continue
        if entrada in letras_tentadas:
            print(f"Você já tentou a letra '{entrada}'. Tente outra.")
            continue
        return entrada

def atualizar_palavra_oculta(palavra, palavra_oculta, letra):
    """
    Atualiza palavra_oculta com a letra informada, se esta estiver na palavra.
    Retorna True se a letra estiver na palavra, False caso contrário.
    """
    acertou = False
    for idx, char in enumerate(palavra):
        if char == letra:
            palavra_oculta[idx] = letra
            acertou = True
    return acertou

def jogo_da_forca():
    """
    Função principal do jogo da forca.
    Executa o loop do jogo até condição de vitória ou derrota.
    Ao final, pergunta se o jogador deseja jogar novamente.
    """
    print("=== Jogo da Forca ===")

    while True:
        palavra = escolher_palavra(PALAVRAS)
        palavra_oculta = inicializar_espaco_oculto(palavra)
        tentativas_restantes = 6
        letras_tentadas = set()

        # Loop do jogo
        while tentativas_restantes > 0 and "_" in palavra_oculta:
            mostrar_estado(palavra_oculta, tentativas_restantes, letras_tentadas)
            letra = obter_letra_valida(letras_tentadas)
            letras_tentadas.add(letra)

            if atualizar_palavra_oculta(palavra, palavra_oculta, letra):
                print(f"Boa! A letra '{letra}' está na palavra.")
            else:
                tentativas_restantes -= 1
                print(f"Que pena! A letra '{letra}' NÃO está na palavra.")

        # Fim da rodada: vitória ou derrota
        if "_" not in palavra_oculta:
            print("\nParabéns! Você venceu!")
        else:
            print("\nVocê perdeu!")

        print(f"A palavra era: '{palavra}'")

        # Perguntar se deseja jogar novamente
        while True:
            resposta = input("\nDeseja jogar novamente? (s/n): ").strip().lower()
            if resposta in ("s", "sim"):
                print("\nIniciando um novo jogo...")
                break
            elif resposta in ("n", "nao", "não"):
                print("Obrigado por jogar! Até a próxima.")
                return
            else:
                print("Resposta inválida. Digite 's' para sim ou 'n' para não.")

if __name__ == "__main__":
    jogo_da_forca()