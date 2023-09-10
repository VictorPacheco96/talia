import cv2 #PERMITE ACESSAR A WEBCAM - Ao instalar o mediapipe, ele já vem atrelado
import mediapipe as mp #FAZ A PARTE DE RECONHECER CORPO, ROSTO E MÃOS E DEFINIR A LOCALIZAÇÃO DE MARCADORES COM BASE NESSE RECONHECIMENTO
import numpy as np
import pyttsx3
from keras import models
from Functions import mediapipe_detection as md
from Functions import draw_landmarks as dl
from Functions import extract_keypoints as ek
from Functions import probability_bar as pb
import lista_gestos as lg


def ler_gestos() :
    #Cores das barras das 5 palavras mais prováveis
    colors = [
        (245,117,16),
        (117,245,16),
        (16,117,245),
        (50,245,50),
        (150,245,150)
    ]

    # Carregando modelo treinado
    model = models.load_model("gestos.keras")

    # Iniciando voz
    engine = pyttsx3.init()

    #Variáveis do mediapipe
    mp_holistic = mp.solutions.holistic #Possui modelos de marcadores do corpo, mãos e rosto
    mp_drawing = mp.solutions.drawing_utils #Desenha esses marcadores na tela

    #Variáveis de detecção
    sequence = [] #Vai conter os 30 frames
    sentence = [] #Vai conter um histórico de traduções feitas
    threshold = 0.9 #Taxa mínima de confiança necessária no resultado para o resultado ser exibido
    actions = np.array(lg.lista_nomes)
    res = [0]

    # Instanciando a câmera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Captura", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Captura", 1080, 720)

    # Acessando modelo Holistic do mediapipe
    # [!!!testar esses dois parametros depois]
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Executa enquanto a webcam está ativa
        while cap.isOpened():
            # Coletando o conteúdo capturado pela câmera
            # Frame é a imagem capturada em si
            # ret é só um boooleano
            ret, frame = cap.read()

            # Fazendo detecções
            image, results = md.mediapipe_detection(frame, holistic)
            dl.draw_landmarks(image, results)

            #2) Lógica da predição
            keypoints = ek.extract_keypoints(results)
            sequence.append(keypoints)

            # garantindo que o sequence conterá sempre apenas os últimos 30 frames coletados
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                #print(res0)

                #res = res0[0]


            ##### 3) Visualizando resultado #####
            #Checa se o grau de confiança da tradução é maior que 0.4
            if res[np.argmax(res)] > threshold and actions[np.argmax(res)][0] != "_" :
                # Caso ainda não existam palavras no array, é adicionado dentro do else
                if len(sentence) > 0:
                #    # É checado se a palavra atual é igual a palavra anterior, se sim é ignorado a tradução para evitar repetição
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                        engine.say(actions[np.argmax(res)])
                        engine.runAndWait() #DIZENDO A TRADUÇÃO
                        sequence = [] #Zerando sequencia de frames após a tradução
                else:
                    sentence.append(actions[np.argmax(res)])
                    engine.say(actions[np.argmax(res)])
                    engine.runAndWait() #DIZENDO A TRADUÇÃO
                    sequence = []  # Zerando sequencia de frames após a tradução

            # Garantindo que só serão exibidas as últimas 5 palavras traduzidas
            if len(sentence) > 5:
                sentence = sentence[-5:]

            #Pegando o nome das 5 ações mais prováveis
            mais_provaveis = []
            print(f"res: {res}")
            previsao = res.copy()
            print(f"previsao: {previsao}")

            print(f"type: {type(previsao)}")
            if 0 not in previsao :
                for i in range(0,5):
                    mais_provaveis.append(actions[np.argmax(previsao)]) #INSERINDO O 1° MAIS PROVÁVEL
                    previsao = np.delete(previsao,np.argmax(previsao)) #.pop(np.argmax(previsao)) #APAGANDO O 1° MAIS PROVÁVEL

                #Exibindo as barras de probabilidade das 5 palavras mais prováveis
                image = pb.probability(res[:5], mais_provaveis, image, colors)

            # Exibindo as 5 últimas traduções na tela
            cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Esse é o comando que mostra na tela o frame capturado
            # "Captura" é o nome da telinha que abre
            # Essa telinha trava caso o loop for interrompido
            # Sempre que for usar esse comando parece ser bom fazer dentro de um loop igual a esse
            # Mesmo que seja exibido uma imagem fixa
            # TESTADO: Mesmo que aja outro loop em seguida a telinha também para de responder
            # Então não é culpa do término da execução do código em si e sim do término do loop
            cv2.imshow("Captura", image)

            # Encerra o loop quando é apertado a tecla "q"
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()  # Para de monitorar a câmera
        cv2.destroyAllWindows()  # Fecha todas as telas do cv2.imshow(), mesmo que elas não estejam respondendo
