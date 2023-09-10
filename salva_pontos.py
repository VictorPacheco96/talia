import cv2 #PERMITE ACESSAR A WEBCAM - Ao instalar o mediapipe, ele já vem atrelado
import mediapipe as mp #FAZ A PARTE DE RECONHECER CORPO, ROSTO E MÃOS E DEFINIR A LOCALIZAÇÃO DE MARCADORES COM BASE NESSE RECONHECIMENTO
import os #AJUDA A TRABALHAR COM PASTAS E ENDEREÇOS DE PASTA DO PRÓPRIO COMPUTADOR
import numpy as np #AJUDA A TRABALHAR COM DIFERENTES ARRAYS
from Functions import mediapipe_detection as md
from Functions import draw_landmarks as dl
from Functions import extract_keypoints as ek

#Variáveis do mediapipe
mp_holistic = mp.solutions.holistic #Possui modelos de marcadores do corpo, mãos e rosto
mp_drawing = mp.solutions.drawing_utils #Desenha esses marcadores na tela

# Coletando nomes dos gestos que cujo as imagens serão capturadas
gestos_nomes = []
gesto = ""
while gesto != "exit" :
    gesto = input("Digite o nome de um gesto que deseja detectar: ")
    gestos_nomes.append(gesto) if gesto != "exit" else ""

#DATA_PATH é uma pasta que vai guardar amostras de gestos salvos como arrays do numpy
DATA_PATH = os.path.join('Gestos') #Pasta que vai guardar todos os gestos
actions = np.array(gestos_nomes) #Cada String aqui é uma pasta, que representa 1 sinal específico da LIBRAS
nro_amostra_inicial = input("A primeira amostra que você ensinar deve ser identificada por qual número? Digite um número de 0 a infinito: ")
no_sequences = input("Quantas amostras de cada gesto você vai ensinar? ")
sequence_length = 30 #Números de frames por amostra

# Criando pastas
for action in actions:
    for sequence in range(int(nro_amostra_inicial),(int(nro_amostra_inicial) + int(no_sequences)*2)):
        try:
            os.makedirs(os.path.join(DATA_PATH,action, str(sequence)))
        except:
            pass

# Instanciando a câmera
cap = cv2.VideoCapture(0)
cv2.namedWindow("Captura", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Captura", 1080, 720)

# Acessando modelo Holistic do mediapipe
# Testado outros valores para o min_detection_confidence e min_tracking_confidence, melhor deixar 0.5 para os 2 por hora
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #Se eu uso o mesmo holistic para imagens normais e para imagens invertidas, o holistic se perde, porque ele fica tentando acompanhar seu movimento e acaba passando por duas imagens totalmente diferentes
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_rh_v:

        # Loop de sinais
        for action in actions:
            #DELAY ANTES DE COMEÇAR A ENSINAR UM NOVO SINAL
            for x in reversed(range(10)):
                success, image = cap.read()
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f'Em {x} segundos: {action}', (15, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("Captura", image)
                cv2.waitKey(1000)

            # Loop de amostras ("Vídeos")
            no_sequences_rh_v = int(no_sequences) + int(nro_amostra_inicial) #sequencias das amostras invertidas na horizontal

            for sequence in range(int(nro_amostra_inicial),(int(nro_amostra_inicial) + int(no_sequences))):
                # Loop de frames
                for frame_num in range(sequence_length):
                    # Coletando o conteúdo capturado pela câmera
                    # Frame é a imagem capturada em si
                    # success é só um boooleano que indica se a câmera foi detectada com sucesso ou não
                    # o read() precisa está dentro desse for para fazer uma captura por frame_num de cada sequence de cada action
                    success, frame = cap.read()

                    # Parando o código caso a câmera não foi detectada
                    if not success:
                        print("Câmera não detectada!")
                        exit()

                    # Fazendo detecções dos pontos
                    image_h_v, results_h_v = md.mediapipe_detection(frame, holistic) #amostras originais
                    image_rh_v, results_rh_v = md.mediapipe_detection(cv2.flip(frame, 1), holistic_rh_v) #amostras invertidas na horizontal

                    # Inserindo pontos na imagem
                    dl.draw_landmarks(image_h_v, results_h_v)

                    # Legendas que indicam qual amostra e gesto está sendo coletado
                    # UTIL PARA SABER QUANDO COMEÇA E QUANDO TERMINA CADA COLETA
                    if frame_num == 0:
                        cv2.rectangle(image_h_v, (0, 0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image_h_v, 'INICIANDO COLETA', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(image_h_v, f'Demonstrando "{action}" -- amostra numero {sequence} -- {frame_num}', (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.waitKey(
                            2000)  # Mandando esperar para ter tempo de você voltar pra posição original antes de começar a próxima coleta
                    else:
                        cv2.rectangle(image_h_v, (0, 0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image_h_v, f'Demonstrando "{action}" -- amostra numero {sequence} -- {frame_num}', (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # Montando array de coordenadas
                    keypoints_h_v = ek.extract_keypoints(results_h_v) #amostras originais
                    keypoints_rh_v = ek.extract_keypoints(results_rh_v) #amostras invertidas na horizontal

                    # Pasta onde o frame será salvo
                    npy_path_h_v = os.path.join(DATA_PATH, action, str(sequence),str(frame_num)) #amostras originais
                    npy_path_rh_v = os.path.join(DATA_PATH, action, str(no_sequences_rh_v), str(frame_num)) #amostras invertidas na horizontal

                    #Salvando pontos
                    np.save(npy_path_h_v, keypoints_h_v) #amostras originais
                    np.save(npy_path_rh_v, keypoints_rh_v) #amostras invertidas na horizontal

                    # Esse é o comando que mostra na tela o frame capturado
                    # "Captura" é o nome da telinha que abre
                    # Essa telinha trava caso o loop for interrompido
                    # Sempre que for usar esse comando parece ser bom fazer dentro de um loop, mesmo que seja exibido uma imagem fixa
                    # TESTADO: Mesmo que aja outro loop após o termino desse a telinha também para de responder
                    # A janela com imagem travar não é culpa do término da execução do código em si e sim do término do loop
                    cv2.imshow("Captura", image_h_v)

                    # Encerra o loop quando é apertado a tecla "q"
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                # Aumentando no_sequences das amostras viradas
                no_sequences_rh_v = no_sequences_rh_v + 1  # amostras invertidas na horizontal

        cap.release()  # Para de monitorar a câmera
        cv2.destroyAllWindows()  # Fecha todas as telas do cv2.imshow(), mesmo que elas não estejam respondendo