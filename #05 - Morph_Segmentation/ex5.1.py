import cv2

image = cv2.imread("../images/wdg2.bmp", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('../images/mon1.bmp', cv2.IMREAD_GRAYSCALE)


if image is None or image2 is None:
    print("Erro no caminho da imagem")
    exit(-1)

img_binary = image.copy()
_, img_binary = cv2.threshold(img_binary, 120, 255, cv2.THRESH_BINARY_INV)

kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

# cv2.imshow("Original", image)
# cv2.imshow("Binario Inverso", img_binary)

# 5.1
'''
dilation = cv2.dilate(img_binary, kernel_square, iterations=1)
cv2.imshow("Dilation", dilation)
'''

# 5.2
'''
erosion = cv2.erode(img_binary, kernel_square, iterations=5, anchor=(0,1))
cv2.imshow("Erosion", erosion)
'''

# 5.3
'''
img_binary2 = image2.copy()
_, img_binary2 = cv2.threshold(img_binary2, 90, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Binario Inverso", img_binary2)

erosion = cv2.erode(img_binary2, kernel_circle, iterations=2)
cv2.imshow("Erosion", erosion)
erosion = cv2.erode(img_binary2, kernel_square, iterations=2)
cv2.imshow("Erosion Square", erosion)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()