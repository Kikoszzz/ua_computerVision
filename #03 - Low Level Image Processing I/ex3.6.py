import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../images/deti.bmp', cv2.IMREAD_UNCHANGED)

if image is None:
    print("ERRO CAMINHO IMAGEM")
    exit(-1)

if len(image.shape) < 3:
    print("ERRO IMAGEM NAO RGB")
    exit(-1)

copy = image.copy()

b, g, r = cv2.split(copy)

gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

hist_b = cv2.calcHist([b], [0], None, [256], [0,256])
hist_g = cv2.calcHist([g], [0], None, [256], [0,256])
hist_r = cv2.calcHist([r], [0], None, [256], [0,256])
hist_gray = cv2.calcHist([gray], [0], None, [256], [0,256])

cv2.namedWindow("Original")
cv2.namedWindow("Gray")

cv2.imshow("Original", image)
cv2.imshow("Gray", gray)

plt.figure("Histogramas")

plt.subplot(2,2,1)
plt.title("Blue")
plt.plot(hist_b)

plt.subplot(2,2,2)
plt.title("Green")
plt.plot(hist_g)

plt.subplot(2,2,3)
plt.title("Red")
plt.plot(hist_r)

plt.subplot(2,2,4)
plt.title("Gray")
plt.plot(hist_gray)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()