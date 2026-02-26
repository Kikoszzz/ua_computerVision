import cv2

bmp = cv2.imread(r".\deti.bmp", cv2.IMREAD_UNCHANGED)

if bmp is None:
    print("Image file could not be opened")
    exit(-1)

img_gray = cv2.cvtColor(bmp, cv2.COLOR_BGR2GRAY)

cv2.imshow("Cor", bmp)
cv2.imshow("Preto e Branco", img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()