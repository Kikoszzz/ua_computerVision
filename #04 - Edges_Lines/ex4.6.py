import sys
import numpy as np
import cv2

imagePath = "../images/Bikesgray.jpg"

image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

if image is None:
	print("Image file could not be open!")
	exit(-1)

# Show original
cv2.imshow('Orginal', image)

blur = cv2.GaussianBlur(image, (5, 5), 1.2)
threshold1 = 50
threshold2 = 150
edges = cv2.Canny(blur, threshold1, threshold2, apertureSize=3, L2gradient=False)

cv2.namedWindow("Canny", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Canny", edges)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)

imageLines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
if lines is not None:
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(imageLines, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.namedWindow("Hough Lines P", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Hough Lines P", imageLines)

cv2.waitKey(0)
cv2.destroyAllWindows()

