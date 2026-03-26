import cv2
import numpy as np
import math

print('Choose image mode:')
print('1 - default (current images)')
print('2 - alternative (gui.JPG)')
choice = input('Option [1/2]: ').strip()

src_path = '../images/gui.JPG' if choice == '2' else '../images/lena.jpg'
dst_path = '../images/gui_tf.jpg' if choice == '2' else '../images/lena_tf.jpg'

src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
dst = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)

if src is None or dst is None:
    print(f'Image not found. Run ex6.1.py first to create {dst_path}')
    exit(-1)

if choice == '2' and src is not None:
    f = 600 / src.shape[1]
    src = cv2.resize(src, None, fx=f, fy=f)
    dst = cv2.resize(dst, None, fx=f, fy=f)

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

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

if len(src_pts) < 3:
    print('Not enough matches to estimate affine transform')
    exit(-1)

M, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)

warp_dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))
diff = cv2.absdiff(warp_dst, dst)

print('\nEstimated affine matrix from automatic correspondences:')
print(M)

a, c, tx = M[0]
b, d, ty = M[1]

sx = math.copysign(math.sqrt(a * a + b * b), a)
sy = math.copysign(math.sqrt(c * c + d * d), d)
psi = math.degrees(math.atan2(b, a))

print('\nEstimated parameters:')
print(f'tx = {tx:.4f}')
print(f'ty = {ty:.4f}')
print(f'sx = {sx:.4f}')
print(f'sy = {sy:.4f}')
print(f'rotation (deg) = {psi:.4f}')

cv2.imshow('Automatic matches', cv2.drawMatches(src, kp1, dst, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
cv2.imshow('Warped source', warp_dst)
cv2.imshow('Difference |warped - transformed|', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
