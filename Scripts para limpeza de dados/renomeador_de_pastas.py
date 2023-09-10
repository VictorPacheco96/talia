#Código para trocar nome de pastas de amostras quando for necessário juntar amostras que possuem o mesmo número

import os

nome_gesto = input("Nome da pasta do gesto: ")
nro_amostras = int(input("Número de amostras contidas nela: "))
nro_inicio_atual = int(input("Número atual da primeira amostra: "))
nro_inicio = int(input("Número que deve possuir a primeira amostra após a edição: "))

DATA_PATH = os.path.join('Novos/Gestos_sala',nome_gesto)

for amostra in range(nro_inicio_atual,(nro_inicio_atual + nro_amostras)) :
    amostra_pasta = os.path.join(DATA_PATH, str(amostra))
    amostra_novoNome = os.path.join(DATA_PATH, str(nro_inicio))
    os.rename(amostra_pasta, amostra_novoNome)

    nro_inicio = nro_inicio + 1