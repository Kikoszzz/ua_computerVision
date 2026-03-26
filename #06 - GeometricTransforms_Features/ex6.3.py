import cv2

print('Choose image mode:')
print('1 - default (current images)')
print('2 - alternative (gui.JPG)')
choice = input('Option [1/2]: ').strip()

if choice == '2':
    src_path = '../images/gui.JPG'
    dst_path = '../images/gui_tf.jpg'
else:
    src_path = '../images/lena.jpg'
    dst_path = '../images/lena_tf.jpg'

src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
dst = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)

if src is None or dst is None:
    print(f'Image not found. Run ex6.1.py first to create {dst_path}')
    exit(-1)

if src.shape != dst.shape:
    dst = cv2.resize(dst, (src.shape[1], src.shape[0]))
    print('Adjusted transformed image size to match original image.')

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(src, None)
kp2, des2 = sift.detectAndCompute(dst, None)

src_kp = cv2.drawKeypoints(src, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst_kp = cv2.drawKeypoints(dst, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(f'Keypoints in original image: {len(kp1)}')
print(f'Keypoints in transformed image: {len(kp2)}')

cv2.imshow('Original keypoints', src_kp)
cv2.imshow('Transformed keypoints', dst_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
