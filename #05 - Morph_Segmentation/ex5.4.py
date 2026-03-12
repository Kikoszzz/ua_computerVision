import cv2

image = cv2.imread('../images/art4.bmp', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Erro no caminho da imagem")
    exit(-1)

kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22,22))

kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3,9))

open = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_vert)

close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_circle)

cv2.imshow("Original", image)
#cv2.imshow("Open", open)
cv2.imshow("Close", close)

cv2.waitKey(0)
cv2.destroyAllWindows()