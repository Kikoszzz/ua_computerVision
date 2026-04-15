import os
import glob
import numpy as np
import cv2

"""
ex7.4.py

Carrega parâmetros intrínsecos/distortion de um arquivo NPZ e realiza calibração
externa (posição/rotação) para cada imagem da sequência usando `solvePnP`.

Uso:
 - Tenha um arquivo de parâmetros: `camera_params.npz` ou `camera_7.3.npz` ou `camera.npz`.
 - Coloque as imagens em ../images/left*.jpg ou modifique a pattern.

O script imprime `intrinsics` e `distortion` e então, para cada imagem que
contiver o padrão, calcula e imprime `rvec` e `tvec`. Salva os resultados em
`external_7.4.npz`.
"""

# Parâmetros do tabuleiro (número de cantos internos)
BOARD_W = 7
BOARD_H = 7
# Tamanho do quadrado em metros (mesma unidade usada na calibração intrínseca)
SQUARE_SIZE = 0.035

# Procurar arquivo de parâmetros intrínsecos
npz_candidates = ['camera_params.npz', 'camera_7.3.npz', 'camera.npz']
found_npz = None
for p in npz_candidates:
    if os.path.exists(p):
        found_npz = p
        break

if found_npz is None:
    print('Nenhum arquivo de parâmetros intrínsecos encontrado. Gere um com ex7.3.py ou chessboard.py.')
    exit()

with np.load(found_npz) as data:
    intrinsics = data['intrinsics']
    distortion = data['distortion']

print('Loaded intrinsics from', found_npz)
print('Intrinsics:')
print(intrinsics)
print('\nDistortion:')
print(distortion)

# Preparar objeto 3D do padrão (z=0)
objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE

# Procurar imagens
images = sorted(glob.glob('../images/left*.jpg'))
if len(images) == 0:
    print('Nenhuma imagem encontrada em ../images/left*.jpg')
    # não sair ainda — podemos usar modo câmera ao vivo se desejar
    images = []

DETECCION_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

rvecs_list = []
tvecs_list = []
img_names = []

# Tentar múltiplos tamanhos/combinações comuns (width, height) de cantos internos
candidate_sizes = [(6, 9), (9, 6), (7, 7), (6, 7), (7, 6)]

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = False
    detected_size = None
    corners = None
    for sz in candidate_sizes:
        ret, c = cv2.findChessboardCorners(gray, sz, DETECCION_FLAGS)
        if ret:
            found = True
            detected_size = sz
            corners = c
            break

    if not found:
        print(f'Padrão não detectado em {os.path.basename(fname)} (tentadas: {candidate_sizes})')
        continue

    print(f'Detected pattern size {detected_size} in {os.path.basename(fname)}')

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Preparar objp de acordo com o tamanho detectado
    objp_detect = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
    objp_detect[:, :2] = np.mgrid[0:detected_size[0], 0:detected_size[1]].T.reshape(-1, 2) * SQUARE_SIZE

    # solvePnP para obter rotações e translações externas (por imagem)
    success, rvec, tvec = cv2.solvePnP(objp_detect, corners2, intrinsics, distortion)
    if not success:
        print(f'solvePnP falhou para {os.path.basename(fname)}')
        continue

    print(f'Image: {os.path.basename(fname)}')
    print('rvec:')
    print(rvec.ravel())
    print('tvec:')
    print(tvec.ravel())
    print('---')

    rvecs_list.append(rvec.ravel())
    tvecs_list.append(tvec.ravel())
    img_names.append(os.path.basename(fname))

if len(rvecs_list) == 0 and len(images) == 0:
    print('Nenhum resultado externo calculado a partir de imagens estáticas.')

# --- Modo câmera ao vivo: detectar padrão, compute solvePnP e projetar cubo ao vivo ---
def draw_cube(img, imgpts):
    pts = imgpts.reshape(-1, 2)
    pts = pts.astype(int)
    # base
    cv2.drawContours(img, [pts[:4]], -1, (0, 255, 0), 2)
    # topo
    cv2.drawContours(img, [pts[4:]], -1, (0, 0, 255), 2)
    # linhas verticais
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), (255, 0, 0), 2)

MODE = 'camera'  # 'images' or 'camera'

if MODE == 'camera':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Não foi possível abrir a câmera.')
    else:
        print('Modo câmera: detectando padrão e projetando cubo. Pressione C para salvar pose, Q para sair.')
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detectar padrão tentando candidate_sizes
            found = False
            detected_size = None
            corners = None
            for sz in candidate_sizes:
                retf, c = cv2.findChessboardCorners(gray, sz, DETECCION_FLAGS)
                if retf:
                    found = True
                    detected_size = sz
                    corners = c
                    break

            if found:
                cv2.drawChessboardCorners(display, detected_size, corners, True)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objp_detect = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
                objp_detect[:, :2] = np.mgrid[0:detected_size[0], 0:detected_size[1]].T.reshape(-1, 2) * SQUARE_SIZE

                success, rvec, tvec = cv2.solvePnP(objp_detect, corners2, intrinsics, distortion)
                if success:
                    # definir cubo em torno da origem do padrão
                    cube_size = SQUARE_SIZE * 3.0
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
                    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, intrinsics, distortion)
                    draw_cube(display, imgpts)

            cv2.imshow('external_calib_live', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c') and found and success:
                rvecs_list.append(rvec.ravel())
                tvecs_list.append(tvec.ravel())
                img_names.append('camera_frame_%d' % saved_count)
                saved_count += 1
                print(f'Pose salva #{saved_count}')

        cap.release()
        cv2.destroyAllWindows()

# Salvar resultados externos (provenientes de imagens estáticas e/ou capturas de câmera)
if len(rvecs_list) > 0:
    np.savez('external_7.4.npz', images=img_names, rvecs=np.array(rvecs_list), tvecs=np.array(tvecs_list))
    print('Saved external_7.4.npz with external parameters for processed images.')
