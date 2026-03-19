# Aula_04_ex_05.py
#
# Canny edge detector with a simple CLI image menu
#

import os
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


def resolveImagePath(filename):
	base_dir = os.path.dirname(__file__)
	local_path = os.path.join(base_dir, filename)
	if os.path.exists(local_path):
		return local_path

	return os.path.join(base_dir, "..", "images", filename)


def chooseImage():
	image_options = ["wdg2.bmp", "lena.jpg", "cln1.bmp", "Bikesgray.jpg"]

	print("Escolhe a imagem para testar o Canny detector:")
	for idx, image_name in enumerate(image_options, start=1):
		print(f"{idx}. {image_name}")

	while True:
		choice = input("Opcao (1-4): ").strip()
		if choice in ("1", "2", "3", "4"):
			return image_options[int(choice) - 1]

		print("Opcao invalida. Tenta novamente.")


selected_image = chooseImage()
image_path = resolveImagePath(selected_image)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

print("Imagem selecionada:", selected_image)
printImageFeatures(image)

cv2.imshow("Orginal", image)

# Canny edge detector
edges = cv2.Canny(image, 100, 150)
cv2.namedWindow("Canny", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Canny", edges)

print("Pressiona qualquer tecla para fechar.")
cv2.waitKey(0)
cv2.destroyAllWindows()


