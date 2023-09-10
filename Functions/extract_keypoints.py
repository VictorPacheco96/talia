import numpy as np #AJUDA A TRABALHAR COM DIFERENTES ARRAYS

# Formata o resulto da detecção retornando um único array, com 3 arrays, um com todas as coordenadas de todos os pontos das mãos, outro igual para o corpo e outro para o rosto
def extract_keypoints(results):
    # Coletando coordenadas de todos os pontos da pose e depois juntando tudo em um único array
    # Depois do flatten(), todos os campos que antes pertenciam a diferentes pontos, agora estão embrulhados no mesmo array, porém ainda mantendo a ordenação
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(44 * 3)

    # Coletando coordenadas de todos os pontos da mão esquerda e depois juntando tudo em um único array
    # Se essa mão não for detectada, é preenchido um array de 63 zeros
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Coletando coordenadas de todos os pontos da mão direita e depois juntando tudo em um único array
    # Se essa mão não for detectada, é preenchido um array de 63 zeros
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    '''
    # Coletando coordenadas de todos os pontos do rosto e depois juntando tudo em um único array
    # Se o rosto não for detectado, é preenchido um array de 1404 zeros
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    '''

    return np.concatenate([pose, lh, rh])