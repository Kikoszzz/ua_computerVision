import sys
import numpy as np
import cv2

def printImageFeatures(image):
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)


image = cv2.imread("../images/Bikesgray.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

# Show original
cv2.imshow('Orginal', image)

# Sobel Operator 3 x 3
'''
imageSobel3x3_X = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
imageSobel3x3_Y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

cv2.namedWindow("Sobel 3 x 3 - X", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Sobel 3 x 3 - X", imageSobel3x3_X)
cv2.namedWindow("Sobel 3 x 3 - Y", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Sobel 3 x 3 - Y", imageSobel3x3_Y)

# Convert result back to numpy for absolute and 8-bit conversion
image8bits = cv2.convertScaleAbs(imageSobel3x3_X)
image8bitsY = cv2.convertScaleAbs(imageSobel3x3_Y)
cv2.imshow("8 bits - Sobel 3 x 3 - X", image8bits)
cv2.imshow("8 bits - Sobel 3 x 3 - Y", image8bitsY)
'''

# canny

edges1 = cv2.Canny(image, 1, 255, apertureSize=3, L2gradient=False)
edges2 = cv2.Canny(image, 220, 225, apertureSize=3, L2gradient=False)
edges3 = cv2.Canny(image, 1, 128, apertureSize=3, L2gradient=False)


cv2.namedWindow("Canny 1-255", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Canny 1-255", edges1)

cv2.namedWindow("Canny 220-225", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Canny 220-225", edges2)

cv2.namedWindow("Canny 1-128", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Canny 1-128", edges3)

# camera
'''
capture = cv2.VideoCapture(0)

if not capture.isOpened():
	print("Could not open camera.")
	exit(-1)

threshold1 = 1
threshold2 = 255

while True:
	ret, frame = capture.read()
	if not ret:
		print("Failed to read frame from camera.")
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

	sobel_x_video = cv2.convertScaleAbs(sobel_x)
	sobel_y_video = cv2.convertScaleAbs(sobel_y)
	
	canny = cv2.Canny(gray, threshold1, threshold2, apertureSize=3, L2gradient=False)

	cv2.imshow("video", frame)
	cv2.imshow("video - sobel x", sobel_x_video)
	cv2.imshow("video - sobel y", sobel_y_video)
	cv2.imshow("video - canny", canny)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

capture.release()
'''

cv2.waitKey(0)
cv2.destroyAllWindows()