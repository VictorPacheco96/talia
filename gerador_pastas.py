import os #AJUDA A TRABALHAR COM PASTAS E ENDEREÇOS DE PASTA DO PRÓPRIO COMPUTADOR
import numpy as np #AJUDA A TRABALHAR COM DIFERENTES ARRAYS

# Coletando nomes dos gestos que cujo as imagens serão capturadas
gestos_nomes = []
gesto = ""
while gesto != "exit" :
    gesto = input("Digite o nome de um gesto que deseja detectar: ")
    gestos_nomes.append(gesto) if gesto != "exit" else ""

#DATA_PATH é uma pasta que vai guardar amostras de gestos salvos como arrays do numpy
DATA_PATH = os.path.join('Gestos') #Pasta que vai guardar todos os gestos
actions = np.array(gestos_nomes) #Cada String aqui é uma pasta, que representa 1 sinal específico da LIBRAS
no_sequences = 30 #Número de amostras por gesto
sequence_length = 45 #Números de frames por amostra

# Criando pastas
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action, str(sequence)))
        except:
            pass
