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

# Read the image from argv
#image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );
image = cv2.imread( "../images/sta2noi1.bmp", cv2.IMREAD_GRAYSCALE );

if  image is None:
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

cv2.imshow('Orginal', image)


# 4.2
#'''
imageAFilter3x3 = cv2.blur( image, (3, 3))
imageAFilter5x5 = cv2.blur( image, (5, 5))
imageAFilter7x7 = cv2.blur( image, (7, 7))

for i in range(3):
	imageAFilter3x3 = cv2.blur( imageAFilter3x3, (3, 3))
	imageAFilter5x5 = cv2.blur( imageAFilter5x5, (5, 5))
	imageAFilter7x7 = cv2.blur( imageAFilter7x7, (7, 7))
#'''

# 4.3
'''
imageAFilter3x3 = cv2.medianBlur( image, 3)
imageAFilter5x5 = cv2.medianBlur( image, 5)
imageAFilter7x7 = cv2.medianBlur( image, 7)

for i in range(3):
	imageAFilter3x3 = cv2.medianBlur( imageAFilter3x3, 3)
	imageAFilter5x5 = cv2.medianBlur( imageAFilter5x5, 5)
	imageAFilter7x7 = cv2.medianBlur( imageAFilter7x7, 7)
'''

# 4.4
'''
imageAFilter3x3 = cv2.GaussianBlur( image, (3, 3), 0)
imageAFilter5x5 = cv2.GaussianBlur( image, (5, 5), 0)
imageAFilter7x7 = cv2.GaussianBlur( image, (7, 7), 0)

for i in range(3):
	imageAFilter3x3 = cv2.GaussianBlur( imageAFilter3x3, (3, 3), 0)
	imageAFilter5x5 = cv2.GaussianBlur( imageAFilter5x5, (5, 5), 0)
	imageAFilter7x7 = cv2.GaussianBlur( imageAFilter7x7, (7, 7), 0)
'''

cv2.namedWindow( "Average Filter 3 x 3", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Average Filter 3 x 3", imageAFilter3x3 )

cv2.namedWindow( "Average Filter 5 x 5", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Average Filter 5 x 5", imageAFilter5x5 )

cv2.namedWindow( "Average Filter 7 x 7", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Average Filter 7 x 7", imageAFilter7x7 )


cv2.waitKey(0)
cv2.destroyAllWindows()