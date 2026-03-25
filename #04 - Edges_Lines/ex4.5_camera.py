import cv2


def main():
	capture = cv2.VideoCapture(0)
	if not capture.isOpened():
		print("Could not open camera.")
		return

	print("Pressiona qualquer tecla para fechar.")
	while True:
		ret, frame = capture.read()
		if not ret:
			print("Could not read frame from camera.")
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 100, 150)

		cv2.imshow("video", frame)
		cv2.imshow("video - canny", edges)

		if cv2.waitKey(1) != -1:
			break

	capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()