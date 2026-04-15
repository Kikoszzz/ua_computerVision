import os
import glob
import cv2
import numpy as np

"""
ex7.3.py
Calibração a partir da câmera do computador ou das imagens fornecidas.

Modo: altere `MODE` para 'camera' ou 'images'.
Pressione 'c' ou Espaço para capturar uma amostra quando usar a câmera.
Pressione 'q' para sair.

Atualize `square_size` com a distância real (em metros ou mm) para obter distâncias métricas.
"""

# Parâmetros do tabuleiro (número de cantos internos)
# Se o seu tabuleiro tem 8x8 quadrados, o número de cantos internos é 7x7.
# Ajuste BOARD_W e BOARD_H para o número de cantos internos do seu padrão.
BOARD_W = 7
BOARD_H = 7
# Tamanho do quadrado em metros (ex.: 0.035 = 35 mm). Atualize para métrica real se quiser distâncias reais.
SQUARE_SIZE = 0.035

# Número de imagens a capturar para calibração
N_IMAGES = 10

# Modo: 'camera' ou 'images'
MODE = 'camera'

# Preparar pontos 3D (z = 0)
objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []  # pontos 3D no mundo
imgpoints = []  # pontos 2D na imagem

DETECCION_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

def process_image_and_collect(fname):
    img = cv2.imread(fname)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), DETECCION_FLAGS)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (BOARD_W, BOARD_H), corners2, ret)
        cv2.imshow('calib', img)
        return True
    else:
        cv2.imshow('calib', img)
        return False

if MODE == 'images':
    images = sorted(glob.glob('../images/left*.jpg'))
    if len(images) == 0:
        print('Nenhuma imagem encontrada em ../images/left*.jpg')
        exit()

    count = 0
    for fname in images:
        found = process_image_and_collect(fname)
        key = cv2.waitKey(500) & 0xFF
        if found:
            count += 1
            print(f'Captured {count}/{N_IMAGES}: {os.path.basename(fname)}')
        if count >= N_IMAGES:
            break

elif MODE == 'camera':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Não foi possível abrir a câmera.')
        exit()

    count = 0
    print('Modo câmera: posicione o tabuleiro e pressione C ou Espaço para capturar. Q para sair.')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Falha ao ler frame da câmera')
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), DETECCION_FLAGS)
        if found:
            cv2.drawChessboardCorners(display, (BOARD_W, BOARD_H), corners, found)

        cv2.putText(display, f'Captured: {count}/{N_IMAGES}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('calib', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') or key == 32:
            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp.copy())
                imgpoints.append(corners2)
                count += 1
                print(f'Captured {count}/{N_IMAGES}')
                if count >= N_IMAGES:
                    break
            else:
                print('Padrão não detectado — ajuste posição/iluminação e tente novamente')

    cap.release()

else:
    print('MODE inválido. Defina MODE = "camera" ou "images"')
    exit()

cv2.destroyAllWindows()

if len(objpoints) == 0:
    print('Nenhum tabuleiro capturado. Saindo.')
    exit()

# Usar o tamanho da última imagem lida para imageSize
# Caso tenha usado câmera, pegamos shape do último frame (frame variável do loop)
try:
    image_size = gray.shape[::-1]
except NameError:
    # se não existir gray (modo images e nenhuma leitura), tentar abrir a primeira imagem
    sample = cv2.imread(sorted(glob.glob('../images/left*.jpg'))[0])
    image_size = sample.shape[1], sample.shape[0]

# Calibração: não passar parâmetros iniciais (None, None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

print('\nErro de reprojeção:', ret)
print('\nIntrinsics:')
print(mtx)
print('\nDistortion:')
print(dist)

# Salvar parâmetros (arquivo específico para este exercício)
np.savez('camera_7.3.npz', intrinsics=mtx, distortion=dist)
print('\nParâmetros salvos em camera_7.3.npz')
