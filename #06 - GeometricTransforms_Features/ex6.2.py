import cv2
import numpy as np
import math

image = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
transformed = cv2.imread('./lena_tf.jpg', cv2.IMREAD_GRAYSCALE)

if image is None or transformed is None:
    print("Image not found")
    exit(-1)

# display copies in color to draw points
image_disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
transf_disp = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)

srcPts = []
dstPts = []

def select_src(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(srcPts) < 3:
        srcPts.append((x, y))
        cv2.circle(image_disp, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(image_disp, str(len(srcPts)), (x+8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imshow('Original', image_disp)

def select_dst(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(dstPts) < 3:
        dstPts.append((x, y))
        cv2.circle(transf_disp, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(transf_disp, str(len(dstPts)), (x+8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('Cópia', transf_disp)

cv2.namedWindow('Original')
cv2.namedWindow('Cópia')
cv2.setMouseCallback('Original', select_src)
cv2.setMouseCallback('Cópia', select_dst)
cv2.imshow('Original', image_disp)
cv2.imshow('Cópia', transf_disp)

print('Selecione 3 pontos na janela "Original" e 3 pontos correspondentes na janela "Cópia".')
print('Pressione ESC para cancelar ou espere as 6 seleções.')

while True:
    k = cv2.waitKey(10) & 0xFF
    if (len(srcPts) >= 3 and len(dstPts) >= 3) or k == 27:
        break

cv2.destroyAllWindows()

if len(srcPts) < 3 or len(dstPts) < 3:
    print('Seleção insuficiente. Abortando.')
    exit(0)

np_src = np.array(srcPts[:3], dtype=np.float32)
np_dst = np.array(dstPts[:3], dtype=np.float32)

M = cv2.getAffineTransform(np_src, np_dst)
warp = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

print('Matriz afim estimada:\n', M)

# Extrai parâmetros
a, c, tx = M[0]
b, d, ty = M[1]
sx = math.copysign(math.sqrt(a*a + b*b), a)
sy = math.copysign(math.sqrt(c*c + d*d), d)
psi_deg = math.degrees(math.atan2(b, a))

print(f'tx={tx:.2f}, ty={ty:.2f}')
print(f'sx={sx:.4f}, sy={sy:.4f}, rotation(deg)={psi_deg:.2f}')

diff = cv2.absdiff(warp, transformed)

cv2.imshow('Warped', cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR))
cv2.imshow('Target', transf_disp)
cv2.imshow('Diff', cv2.applyColorMap(diff, cv2.COLORMAP_JET))
print('Feche as janelas para terminar.')
cv2.waitKey(0)
cv2.destroyAllWindows()
