import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('../images/input.png', cv2.IMREAD_UNCHANGED)

if image is None:
    print("ERRO CAMINHO IMAGEM")
    exit(-1)

if len(image.shape) > 2:
    print("ERRO GRAYSCALE")
    exit(-1)

copy = image.copy()

min_val, max_val, _, _ = cv2.minMaxLoc(copy)

print((min_val, max_val))

copy = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

hist_original = cv2.calcHist([image],[0],None,[256],[0,256])
hist_copy = cv2.calcHist([copy],[0],None,[256],[0,256])

cv2.namedWindow("Original")
cv2.namedWindow("Contrast-Stretching")
cv2.imshow("Original", image)
cv2.imshow("Contrast-Stretching", copy)

plt.figure("Histogramas")

plt.subplot(1, 2, 1)
plt.title("Original")
plt.plot(hist_original)

plt.subplot(1, 2, 2)
plt.title("Contrast-Stretching")
plt.plot(hist_copy)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()