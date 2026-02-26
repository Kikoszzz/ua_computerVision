import numpy as np
import cv2
import sys

bmp = cv2.imread(r".\deti.bmp", cv2.IMREAD_UNCHANGED )
jpg = cv2.imread(r".\deti.jpg", cv2.IMREAD_UNCHANGED )


if bmp is None or jpg is None:
	print("Image file could not be open")
	exit(-1)

diff = cv2.subtract(bmp, jpg)

cv2.imshow( "BMP", bmp )
cv2.imshow( "JPG", jpg )

cv2.imshow("DIFF", diff)

cv2.waitKey( 0 )

cv2.destroyAllWindows()