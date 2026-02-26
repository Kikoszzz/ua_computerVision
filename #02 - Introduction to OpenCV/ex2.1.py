 # Aula_02_ex_01.py
 #
 # Example of image processing with openCV
 # Copy image and modify pixels based on intensity threshold
 #
 # Paulo Dias

#import
import numpy as np
import cv2
import sys


def processar_imagem(original):
	copia = original.copy()
	
	if len(original.shape) == 2: # preto e branco
		for y in range(original.shape[0]):
			for x in range(original.shape[1]):
				if original[y, x] < 128:
					copia[y, x] = 0
	else: # cores
		for y in range(original.shape[0]):
			for x in range(original.shape[1]):
				if np.any(original[y, x] < 128):
					copia[y, x] = 0
	
	return copia

image_file = sys.argv[1]

image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

if image is None or np.shape(image) == ():
	print(f"Image file '{image_file}' could not be opened")
	exit(-1)

print("\n--- Original ---")
print(f"Image Type: {image.dtype}")
print(f"Image Shape: {image.shape}")

copia = processar_imagem(image)

print("\n--- Cópia ---")
print(f"Image Type: {copia.dtype}")
print(f"Image Shape: {copia.shape}")

cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Cópia", cv2.WINDOW_AUTOSIZE)

cv2.imshow("Original", image)
cv2.imshow("Cópia", copia)

cv2.waitKey(0)

cv2.destroyButton("Original Image")
cv2.destroyButton("Modified Image")
