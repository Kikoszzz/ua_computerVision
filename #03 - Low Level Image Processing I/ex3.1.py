import cv2

image = cv2.imread(r"../images/deti.jpg", cv2.IMREAD_UNCHANGED)

if image is None:
    print("Image file could not be open!")
    exit(-1)

is_grayscale = True
line_color = 255

if len(image.shape) > 2:
    is_grayscale = False
    line_color = (128, 128, 128)

if not is_grayscale:
    height, width, channels = image.shape
else:
    height, width = image.shape

copy = image.copy()

spacing = 20

for x in range(0, width, spacing):
    cv2.line(copy, (x, 0), (x, height - 1), line_color, 1)

for y in range(0, height, spacing):
    cv2.line(copy, (0, y), (width - 1, y), line_color, 1)

cv2.imwrite("image_with_grid.jpg", copy)
cv2.namedWindow("Grid Image")
cv2.imshow("Grid Image", copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
