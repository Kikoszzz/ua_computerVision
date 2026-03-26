import cv2
import numpy as np
import math

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
    print(f"Image not found. Run ex6.1.py first to create {dst_path}")
    exit(-1)

if src.shape != dst.shape:
    dst = cv2.resize(dst, (src.shape[1], src.shape[0]))
    print('Adjusted transformed image size to match original image.')

srcPts = []
dstPts = []

def select_src(event, x, y, flags, params):
    global srcPts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(srcPts) < 3:
            srcPts.append((x, y))
            cv2.circle(src_draw, (x, y), 3, 255, 2)
            cv2.putText(src_draw, str(len(srcPts)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.imshow("Original", src_draw)


def select_dst(event, x, y, flags, params):
    global dstPts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dstPts) < 3:
            dstPts.append((x, y))
            cv2.circle(dst_draw, (x, y), 3, 255, 2)
            cv2.putText(dst_draw, str(len(dstPts)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.imshow("Transformed", dst_draw)


src_draw = src.copy()
dst_draw = dst.copy()

cv2.namedWindow("Original")
cv2.namedWindow("Transformed")
cv2.setMouseCallback("Original", select_src)
cv2.setMouseCallback("Transformed", select_dst)

print("Select 3 points in Original and 3 corresponding points in Transformed (same order).")

while True:
    cv2.imshow("Original", src_draw)
    cv2.imshow("Transformed", dst_draw)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:  # ESC
        print("Canceled by user")
        cv2.destroyAllWindows()
        exit(0)

    if len(srcPts) == 3 and len(dstPts) == 3:
        break

np_srcPts = np.array(srcPts).astype(np.float32)
np_dstPts = np.array(dstPts).astype(np.float32)

M = cv2.getAffineTransform(np_srcPts, np_dstPts)

warp_dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))
diff = cv2.absdiff(warp_dst, dst)

print("\nEstimated affine matrix:")
print(M)

a, c, tx = M[0]
b, d, ty = M[1]

sx = math.copysign(math.sqrt(a * a + b * b), a)
sy = math.copysign(math.sqrt(c * c + d * d), d)
psi = math.degrees(math.atan2(b, a))

print("\nEstimated parameters:")
print(f"tx = {tx:.4f}")
print(f"ty = {ty:.4f}")
print(f"sx = {sx:.4f}")
print(f"sy = {sy:.4f}")
print(f"rotation (deg) = {psi:.4f}")

cv2.imshow("Warped source", warp_dst)
cv2.imshow("Difference |warped - transformed|", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()