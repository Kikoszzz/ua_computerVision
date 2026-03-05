import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../images/TAC_PULMAO.bmp', cv2.IMREAD_UNCHANGED)

if image is None:
    print("ERRO CAMINHO IMAGEM")
    exit(-1)

if len(image.shape) > 2:
    print("ERRO GRAYSCALE")
    exit(-1)

copy = image.copy()

copy = cv2.equalizeHist(copy)

hist_original = cv2.calcHist([image], [0], None, [256], [0,256])
hist_copy = cv2.calcHist([copy], [0], None, [256], [0,256])

cv2.namedWindow("Original")
cv2.namedWindow("Histogram Equalization")

cv2.imshow("Original", image)
cv2.imshow("Histogram Equalization", copy)

plt.figure("Histogramas")

plt.subplot(1,2,1)
plt.title("Original")
plt.plot(hist_original)

plt.subplot(1,2,2)
plt.title("Equalized")
plt.plot(hist_copy)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()