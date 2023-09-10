import cv2 #PERMITE ACESSAR A WEBCAM - Ao instalar o mediapipe, ele já vem atrelado

# Fazendo detecções
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converte a imagem do esquema de cores BGR para o esquema RGB

    image.flags.writeable = False  # image deixa de ser editável
    results = model.process(image)  # feito a detecção das partes do corpo com mediapipe
    image.flags.writeable = True  # image volta a ser editável

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # converte o esquema de cores de voltas de RGB para BGR

    return image, results