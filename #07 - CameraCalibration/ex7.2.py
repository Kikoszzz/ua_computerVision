import numpy as np
import cv2
import glob

# Board Size (número de cantos internos)
board_h = 9
board_w = 6

# Tamanho do quadrado em unidades arbitrárias (p. ex. 1.0). Ajuste para unidades reais se souber.
square_size = 1.0

# Critério para melhorar precisão dos cantos (sub-pixel)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar pontos 3D do mundo real (z = 0)
import os
import glob
import numpy as np
import cv2

# Parâmetros do tabuleiro
board_h = 9
board_w = 6
square_size = 1.0

# Preparar pontos 3D do tabuleiro (z=0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2) * square_size

# Carregar intrínsecos e distortion gerados pelo script de calibração
# Procurar primeiro por camera_7.3.npz (gerado por ex7.3), depois por camera.npz
npz_candidates = ['camera_7.3.npz', 'camera.npz']
found_npz = None
for p in npz_candidates:
    if os.path.exists(p):
        found_npz = p
        break
if found_npz is None:
    print('Arquivo camera_7.3.npz ou camera.npz não encontrado. Rode ex7.3.py ou chessboard.py primeiro para gerar intrinsics e distortion.')
    exit()

data = np.load(found_npz)
mtx = data['intrinsics']
dist = data['distortion']
print(f'Usando parâmetros de calibração de: {found_npz}')

# Escolher primeira imagem de calibração para projeção
images = sorted(glob.glob('../images/left*.jpg'))
if len(images) == 0:
    print('Nenhuma imagem encontrada em ../images/left*.jpg')
    exit()

img_fname = images[0]
img = cv2.imread(img_fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar cantos na imagem (apenas para obter correspondência imagem->mundo)
ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
if not ret:
    print('Não foi possível detectar o tabuleiro na imagem:', img_fname)
    exit()

# Refinar cantos
corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

# Calcular rvec e tvec com solvePnP (usa os intrínsecos carregados)
success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
if not success:
    print('solvePnP falhou')
    exit()

# Definir pontos 3D do cubo (mesmas unidades de square_size)
cube_size = square_size * 3.0
axis = np.float32([
    [0, 0, 0],
    [0, cube_size, 0],
    [cube_size, cube_size, 0],
    [cube_size, 0, 0],
    [0, 0, -cube_size],
    [0, cube_size, -cube_size],
    [cube_size, cube_size, -cube_size],
    [cube_size, 0, -cube_size]
])

# Projetar pontos 3D para a imagem
imgpts, _ = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
imgpts = np.int32(imgpts).reshape(-1, 2)

def draw_cube(img, pts):
    cv2.drawContours(img, [pts[:4]], -1, (0, 255, 0), 2)
    cv2.drawContours(img, [pts[4:]], -1, (0, 0, 255), 2)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), (255, 0, 0), 2)

draw_cube(img, imgpts)

out_name = 'ex7.2_projected.jpg'
cv2.imwrite(out_name, img)
print('Saved', out_name, 'using image', img_fname)

cv2.imshow('Projected Cube', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

