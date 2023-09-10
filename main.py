from flask import Flask
from flask import render_template
from flask import Response
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from keras import models
from Functions import mediapipe_detection as md
from Functions import draw_landmarks as dl
from Functions import extract_keypoints as ek
from Functions import probability_bar as pb
import lista_gestos as lg

# Configurações do Flask
app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +
     "haarcascade_frontalface_default.xml")
def generate():
     # Cores das barras das 5 palavras mais prováveis
     colors = [
          (245, 117, 16),
          (117, 245, 16),
          (16, 117, 245),
          (50, 245, 50),
          (150, 245, 150)
     ]

     # Carregando modelo treinado
     model = models.load_model("gestos.keras")

     # Iniciando voz
     engine = pyttsx3.init()

     # Variáveis do mediapipe
     mp_holistic = mp.solutions.holistic  # Possui modelos de marcadores do corpo, mãos e rosto
     mp_drawing = mp.solutions.drawing_utils  # Desenha esses marcadores na tela

     # Variáveis de detecção
     sequence = []  # Vai conter os 30 frames
     sentence = []  # Vai conter um histórico de traduções feitas
     threshold = 0.9  # Taxa mínima de confiança necessária no resultado para o resultado ser exibido
     actions = np.array(lg.lista_nomes)
     res = [0]

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

               # 2) Lógica da predição
               keypoints = ek.extract_keypoints(results)
               sequence.append(keypoints)

               # garantindo que o sequence conterá sempre apenas os últimos 30 frames coletados
               print(np.shape(np.array(sequence)))
               sequence = sequence[-30:]

               if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]

               ##### 3) Visualizando resultado #####
               # Checa se o grau de confiança da tradução é maior que 0.4
               if res[np.argmax(res)] > threshold and actions[np.argmax(res)][0] != "_":
                    # Caso ainda não existam palavras no array, é adicionado dentro do else
                    if len(sentence) > 0:
                         #    # É checado se a palavra atual é igual a palavra anterior, se sim é ignorado a tradução para evitar repetição
                         if actions[np.argmax(res)] != sentence[-1]:
                              sentence.append(actions[np.argmax(res)])
                              engine.say(actions[np.argmax(res)])
                              engine.runAndWait()  # DIZENDO A TRADUÇÃO
                              sequence = []  # Zerando sequencia de frames após a tradução
                    else:
                         sentence.append(actions[np.argmax(res)])
                         engine.say(actions[np.argmax(res)])
                         engine.runAndWait()  # DIZENDO A TRADUÇÃO
                         sequence = []  # Zerando sequencia de frames após a tradução

               # Garantindo que só serão exibidas as últimas 5 palavras traduzidas
               if len(sentence) > 5:
                    sentence = sentence[-5:]

               # Pegando o nome das 5 ações mais prováveis
               mais_provaveis = []
               print(f"res: {res}")
               previsao = res.copy()
               print(f"previsao: {previsao}")

               print(f"type: {type(previsao)}")
               if 0 not in previsao:
                    for i in range(0, 5):
                         mais_provaveis.append(actions[np.argmax(previsao)])  # INSERINDO O 1° MAIS PROVÁVEL
                         previsao = np.delete(previsao, np.argmax(
                              previsao))  # .pop(np.argmax(previsao)) #APAGANDO O 1° MAIS PROVÁVEL

                    # Exibindo as barras de probabilidade das 5 palavras mais prováveis
                    image = pb.probability(res[:5], mais_provaveis, image, colors)

               # Exibindo as 5 últimas traduções na tela
               cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
               cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                           cv2.LINE_AA)

               (flag, encodedImage) = cv2.imencode(".jpg", image)
               if not flag:
                    continue
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
     return render_template("index.html")

@app.route("/video_feed")
def video_feed():
     return Response(generate(),
          mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
     app.run(debug=False)

cap.release()