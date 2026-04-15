import numpy as np
import cv2
import glob

# Board Size (número de cantos internos)
board_h = 9
board_w = 6

# Critério para melhorar precisão dos cantos (sub-pixel)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar pontos 3D do mundo real (z = 0)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays para armazenar pontos
objpoints = []  # 3D
imgpoints = []  # 2D

# Ler imagens
images = sorted(glob.glob('../images/left*.jpg'))

img_size = None

for fname in images:
    img = cv2.imread(fname)
    
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar cantos
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    if ret:
        # Refinar cantos (mais precisão)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners2)

        if img_size is None:
            img_size = gray.shape[::-1]

        # Mostrar resultado
        cv2.drawChessboardCorners(img, (board_w, board_h), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

# Verificação de segurança
if img_size is None or len(objpoints) == 0:
    print("Erro: nenhum tabuleiro detectado nas imagens.")
    exit()

# Calibração
_, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)

print("Intrinsics: ")
print(mtx)
print("Distortion : ")
print(dist)
for i in range(len(tvecs)):
    print ("Translations(%d) : " % i )
    print(tvecs[i])
    print ("Rotation(%d) : " % i )
    print(rvecs[i])

# Salvar intrinsics e distortion em arquivo .npz
np.savez('camera.npz', intrinsics=mtx, distortion=dist )
print('\nSaved camera.npz')

cv2.destroyAllWindows()