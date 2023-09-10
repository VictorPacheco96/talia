import os #AJUDA A TRABALHAR COM PASTAS E ENDEREÇOS DE PASTA DO PRÓPRIO COMPUTADOR
import numpy as np #AJUDA A TRABALHAR COM DIFERENTES ARRAYS
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import lista_gestos as lg

# Coletando nomes dos gestos que cujo as imagens serão capturadas

#DATA_PATH é uma pasta que vai guardar frames de amostras de gestos salvos como arrays do numpy
DATA_PATH = os.path.join('Gestos') #Pasta que vai guardar todos os gestos
actions = np.array(lg.lista_nomes) #Cada String aqui é uma pasta, que representa 1 sinal específico da LIBRAS
no_sequences = 70 #Número de amostras por gesto
sequence_length = 30 #Números de frames por amostra

# Criando um mapa de gestos com index e valor
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], [] #sequences são a entrada(x) e labels são as saídas(y)
for action in actions:
    for sequence in range(no_sequences):
        window = [] #cada windows é uma amostra contendo 30 frames
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            print(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy --  {len(res)}"))
            window.append(res) #cada res é um frame
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int) #define um array de 0s e 1s específico para cada categoria

#separando entradas e saídas de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

#Salvando dados de treino
if os.path.exists("Dados de teste") == False :
    os.makedirs("")
npy_path = os.path.join("Dados de teste\\x_test")  # Pasta onde o frame será salvo
np.save(npy_path, x_test)

npy_path = os.path.join("Dados de teste\\y_test")  # Pasta onde o frame será salvo
np.save(npy_path, y_test)

# Criando e treinando rede neural LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

log_dir = os.path.join('Logs')
earlyStop = EarlyStopping(monitor='loss',restore_best_weights=True, mode='min', start_from_epoch=45, patience=5, min_delta=0.0001)
#tensorB = TensorBoard(log_dir=log_dir)

# Criando camadas
model = Sequential()
#64 lstm units que não necessariamente é o número de camadas
# com return_sequences você está especificando que uma camada precisa retornar algo para a camada seguinte usar
#cada linha dessas é uma camada
#a terceira camada LSTM não retorna nada porque a camada seguinte é uma camada densa e não precisa desse retorno
model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(30,258))) #aqui o número de entradas será 30 * 258
model.add(Dropout(0.2))
model.add(LSTM(units=256, activation='relu', return_sequences=True))
model.add(LSTM(units=128, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=actions.shape[0], activation='softmax'))

'''
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30,1662)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
'''

'''
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
'''

# Após serem definidas as configurações das camadas da rede, ela é compilada
# Segundo o cara do vídeo a loss sendo "categorical_crossentropy" é bom para quando você tem um multiplas clasificações
# Para classificações entre duas opções é possível usar binary_crossentropy
#model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# TREINANDO MODELO
# 2000 epochs pode ser muita coisa para apenas 5 categorias, mas vou manter
model.fit(x_train, y_train, epochs=150, callbacks=[earlyStop])

model.save('gestos.keras')