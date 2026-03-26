import cv2
import numpy as np

img = cv2.imread('../images/homography_1.jpg')

if img is None:
    print('Image not found (check homography_1.jpg)')
    exit(-1)

clicked_pts = []
img_draw = img.copy()


def select_points(event, x, y, flags, params):
    global clicked_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < 4:
        clicked_pts.append((x, y))
        cv2.circle(img_draw, (x, y), 4, (0, 0, 255), 2)
        cv2.putText(img_draw, str(len(clicked_pts)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('Homography source', img_draw)


cv2.namedWindow('Homography source')
cv2.setMouseCallback('Homography source', select_points)

print('Select 4 corners of the book in this order: top-left, top-right, bottom-right, bottom-left.')

while True:
    cv2.imshow('Homography source', img_draw)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # ESC
        print('Canceled by user')
        cv2.destroyAllWindows()
        exit(0)

    if len(clicked_pts) == 4:
        break

src_pts = np.array(clicked_pts, dtype=np.float32)

# Book size: 17.5 x 23.5 cm, scaled to pixels
scale = 20
w = int(17.5 * scale)
h = int(23.5 * scale)

dst_pts = np.array(
    [
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ],
    dtype=np.float32,
)

H, _ = cv2.findHomography(src_pts, dst_pts)
warped = cv2.warpPerspective(img, H, (w, h))

print('\nEstimated homography matrix:')
print(H)

cv2.imshow('Homography source', img_draw)
cv2.imshow('Rectified book', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
