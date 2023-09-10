# Caso um frame estiver vazio, ele é preenchido com o conteúdo do último frame não vazio

import os
import numpy as np
import lista_gestos as lg

for nome_gesto in lg.lista_nomes :
    #nome_gesto = input("Nome da pasta do gesto: ")
    #nro_amostras = int(input("Número de amostras contidas nela: "))
    nro_frames = 30

    DATA_PATH = os.path.join('Gestos',nome_gesto)
    nro_amostras = len(os.listdir(DATA_PATH))

    ultima_amostra_cheia = ""
    for amostra in range(nro_amostras) :
        for frame in range(30):
            frame_path = os.path.join(DATA_PATH,str(amostra),f"{str(frame)}.npy")
            print(frame_path)

            with open(frame_path, 'r') as file:
                file.seek(0, os.SEEK_END)

                try:
                    teste = np.load(frame_path)

                    if (frame != 0):
                        ultima_amostra_cheia = frame_path
                except:
                    if(ultima_amostra_cheia != "" and frame != 0):
                        ultima_amostra_cheia_npy = np.load(ultima_amostra_cheia)
                        np.save(frame_path,ultima_amostra_cheia_npy)
                        print("Salvei: ",frame_path)

                        file.seek(0)  # rebobinar o arquivo
                    else:
                        print("Não consegui salvar esse: ",frame_path)