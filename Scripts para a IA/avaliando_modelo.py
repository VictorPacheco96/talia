from keras import models
import numpy as np #AJUDA A TRABALHAR COM DIFERENTES ARRAYS
import os #AJUDA A TRABALHAR COM PASTAS E ENDEREÇOS DE PASTA DO PRÓPRIO COMPUTADOR
import sys
from keras.utils import to_categorical
from keras import models
from sklearn.model_selection import train_test_split

#Acessando a pasta raiz para poder importar a pasta "Functions"
parent = os.path.realpath(__file__)
sys.path.append(parent.replace('\\Scripts para a IA\\avaliando_modelo.py',''))

from lista_gestos import lista_nomes

#==========================#
# TESTANDO MODELO TREINADO #
#==========================#
#DATA_PATH é uma pasta que vai guardar amostras de gestos salvos como arrays do numpy
DATA_PATH = os.path.join('Gestos') #Pasta que vai guardar todos os gestos
actions = np.array(lista_nomes) #Cada String aqui é uma pasta, que representa 1 sinal específico da LIBRAS
no_sequences = 30 #Número de amostras por gesto
sequence_length = 30 #Números de frames por amostra
sequence_length = 30 #Números de frames por amostra

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], [] #sequences são a entrada(x) e labels são as saídas(y)
for action in actions:
    for sequence in range(no_sequences):
        window = [] #cada windows é uma amostra contendo 30 frames
        window = [] #cada windows é uma amostra contendo 45 frames
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res) #cada res é um frame
        sequences.append(window)
        labels.append(label_map[action])
x = np.array(sequences)
y = to_categorical(labels).astype(int) #define um array de 0s e 1s específico para cada categoria
#separando entradas e saídas de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05)
# Carregando modelo treinado
model = models.load_model("gestos.h5")
res = model.predict(x_test)
#===========================================================#
# 10) AVALIANDO MODELO USANDO MATRIX DE CONFUSÃO E ACURÁCIA #
#===========================================================#
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(x_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
#Imprime uma matriz que exibe a quantidade de cada um dos casos abaixo foi encotrado para cada classificação
#[VERDADEIRO NEGATIVO, FALSO POSITIVO]
#[FALSO NEGATIVO, VERDADEIRO POSITIVO]
print(f"multilabel_confusion_matrix: {multilabel_confusion_matrix(ytrue,yhat)}")
#Imprime uma pontuação que diz o quão boa está a acurácia do modelo, lembrando que acurácia != precisão
print(f"accuracy_score: {accuracy_score(ytrue,yhat)}")