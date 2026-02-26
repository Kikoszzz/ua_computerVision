import numpy as np
import cv2
import sys

bmp = cv2.imread(r".\deti.bmp", cv2.IMREAD_UNCHANGED )

def mouse_handler(event, x, y, flags, params):
    if event == cv2.EVENT_RBUTTONDOWN:
        # print("right click")
        cv2.circle(bmp, (x, y), 20, (0, 0, 255), -1) # img, pos, raio, cor (BGR), preenchido
        cv2.imshow( "BMP", bmp )

if bmp is None:
	print("Image file could not be open")
	exit(-1)

cv2.namedWindow("BMP", cv2.WINDOW_AUTOSIZE)
cv2.imshow( "BMP", bmp )

cv2.setMouseCallback("BMP", mouse_handler)

cv2.waitKey( 0 )

cv2.destroyAllWindows()