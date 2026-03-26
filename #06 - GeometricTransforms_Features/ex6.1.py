import cv2
import numpy as np

print('Choose image mode:')
print('1 - default (current images)')
print('2 - alternative (gui.JPG)')
choice = input('Option [1/2]: ').strip()

src_path = '../images/gui.JPG' if choice == '2' else '../images/lena.jpg'
dst_path = '../images/gui_tf.jpg' if choice == '2' else '../images/lena_tf.jpg'

src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
dst = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)

if src is None:
    print(f"Erro no caminho da imagem: {src_path}")
    exit(-1)

if choice == '2' and src is not None:
    f = 600 / src.shape[1]
    src = cv2.resize(src, None, fx=f, fy=f)
    if dst is not None:
        dst = cv2.resize(dst, None, fx=f, fy=f)

rows, cols = src.shape

M = cv2.getRotationMatrix2D((0, 0), 25, 1)
print("Rotation matrix:")
print(M)
M[0][2] = -50
M[1][2] = 100
print("\nRotation + translation matrix:")
print(M)

tf = cv2.warpAffine(src, M, (cols, rows))

ok = cv2.imwrite(dst_path, tf)

if ok:
    print(f"\nSaved transformed image to: {dst_path}")
else:
    print(f"\nFailed to save transformed image to: {dst_path}")

cv2.imshow("Original", src)
cv2.imshow("Transformed", tf)

cv2.waitKey(0)
cv2.destroyAllWindows()