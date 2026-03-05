import cv2

image = cv2.imread("../images/lena.jpg", cv2.IMREAD_GRAYSCALE)
# Testar com imagem gradiente preto -> branco
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

if image is None:
    print("Invalid image path")
    exit(-1)

# THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO and THRESH_TOZERO_INV
binary = image.copy()
binary_inv = image.copy()
trunc = image.copy()
tozero = image.copy()
tozero_inv = image.copy()

_, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(binary_inv, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(trunc, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(tozero, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv = cv2.threshold(tozero_inv, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow('Original', image)
cv2.imshow('Binary', binary)
cv2.imshow('Binary Inv', binary_inv)
cv2.imshow('Trunc', trunc)
cv2.imshow('Tozero', tozero)
cv2.imshow('Tozero Inv', tozero_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()
