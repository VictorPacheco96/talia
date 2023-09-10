import os

import numpy as np
'''
a = os.path.join('Gestos')
pasta = os.listdir(a)
sizeList = []

for file in pasta :
    sizeList.append(len(os.listdir(os.path.join(a,file))))

for i in range(0, 27):
    index = np.argmin(sizeList)
    print(f"{pasta[index]} -- {min(sizeList)}")  # INSERINDO O 1° MAIS PROVÁVEL
    pasta.pop(index)  # APAGANDO O 1° MAIS PROVÁVEL
    sizeList.pop(index)'''

