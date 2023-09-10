# Código para remover dados de rosto de arquivos .npy que ainda possui esses valores

import os
import numpy as np
from lista_gestos import lista_nomes

#for nome_gesto in lista_nomes :
nome_gesto = input("Nome da pasta do gesto: ")
#nro_amostras = int(input("Número de amostras contidas nela: "))
nro_frames = 30

DATA_PATH = os.path.join('Gestos',nome_gesto)
nro_amostras = len(os.listdir(DATA_PATH))

for amostra in range(nro_amostras) :
    for frame in range(30):
        frame_path = os.path.join(DATA_PATH, str(amostra), f"{str(frame)}.npy")
        frame_value = np.load(frame_path)

        if(len(frame_value) == 1662):
            #132 1536
            novo_valor = np.delete(frame_value,np.s_[132:1536])
            np.save(frame_path,novo_valor)
