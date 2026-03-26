import cv2
import numpy as np

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

if des1 is None or des2 is None:
    print('Could not compute descriptors')
    exit(-1)

bf = cv2.BFMatcher(cv2.DescriptorMatcher_BRUTEFORCE, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

numGoodMatches = max(10, int(len(matches) * 0.1))
matches = matches[:numGoodMatches]

im_matches = cv2.drawMatches(
    src,
    kp1,
    dst,
    kp2,
    matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

print(f'Total matches used: {len(matches)}')
print(f'src_pts shape: {src_pts.shape}')
print(f'dst_pts shape: {dst_pts.shape}')

cv2.imshow('Matches', im_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
