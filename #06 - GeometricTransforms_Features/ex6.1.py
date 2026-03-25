import cv2
import numpy as np

image = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Erro no caminho da imagem")
    exit(-1)

rotation = image.copy()
translation = image.copy()
rot_trans = image.copy()

rows, cols = rotation.shape

rot_M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 45, 1)
rot_dst = cv2.warpAffine(rotation, rot_M, (cols, rows))

trans_M = np.array([[1, 0, 100], [0, 1, 50]], dtype=np.float32)
trans_dst = cv2.warpAffine(translation, trans_M, (cols, rows))

rot_M_3 = np.vstack([rot_M.astype(np.float32), [0.0, 0.0, 1.0]])
trans_M_3 = np.vstack([trans_M.astype(np.float32), [0.0, 0.0, 1.0]])
rot_trans_M_3 = trans_M_3 @ rot_M_3
rot_trans_M = rot_trans_M_3[0:2, :].astype(np.float32)
rot_trans_dst = cv2.warpAffine(rot_trans, rot_trans_M, (cols, rows))

cv2.imshow("Original", image)
cv2.imshow("Rotation", rot_dst)
cv2.imshow("Translation", trans_dst)
cv2.imshow("Rotation and Translation", rot_trans_dst)

cv2.waitKey(0)
cv2.destroyAllWindows()