# Desenhando os marcadores
import mediapipe as mp #FAZ A PARTE DE RECONHECER CORPO, ROSTO E MÃOS E DEFINIR A LOCALIZAÇÃO DE MARCADORES COM BASE NESSE RECONHECIMENTO

def draw_landmarks(image, results):
    # Variáveis do mediapipe
    mp_holistic = mp.solutions.holistic  # Possui modelos de marcadores do corpo, mãos e rosto
    mp_drawing = mp.solutions.drawing_utils  # Desenha esses marcadores na tela

    '''
    # Rosto
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),  # Muda o estilo dos marcadores
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)  # Muda o estilo das conexões
    )
    '''

    # Corpo
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),  # Muda o estilo dos marcadores
        mp_drawing.DrawingSpec(color=(0, 121, 121), thickness=5, circle_radius=1)  # Muda o estilo das conexões
    )

    # Mãos
    hand_circles = mp_drawing.DrawingSpec(color=(256, 20, 0), thickness=5,
                                          circle_radius=1)  # Muda o estilo dos marcadores
    hand_connections = mp_drawing.DrawingSpec(color=(80, 0, 121), thickness=5,
                                              circle_radius=1)  # Muda o estilo das conexões

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_circles,
                              hand_connections)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_connections,
                              hand_circles)