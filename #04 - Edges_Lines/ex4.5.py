# Aula_04_ex_05.py
#
# Sobel Operator
#
# Paulo Dias 

#import
import sys
import numpy as np
import cv2

def printImageFeatures(image):
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1;
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)

# Read the image from argv
#image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE )
image = cv2.imread("./lena.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

# Show original (convert to UMat for type stubs)
cv2.imshow('Orginal', cv2.UMat(image))

# Sobel Operator 3 x 3
# compute on UMat to satisfy type hints and use keyword for ksize
src_umat = cv2.UMat(image)
imageSobel3x3_X = cv2.Sobel(src_umat, cv2.CV_64F, 1, 0, ksize=3)

cv2.namedWindow("Sobel 3 x 3 - X", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Sobel 3 x 3 - X", imageSobel3x3_X)

# Convert result back to numpy for absolute and 8-bit conversion
sobel_np = imageSobel3x3_X.get()
image8bits = np.uint8(np.absolute(sobel_np))
cv2.imshow("8 bits - Sobel 3 x 3 - X", cv2.UMat(image8bits))

cv2.waitKey(0)


