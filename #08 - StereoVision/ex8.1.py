# Cheesboard.py
 #
 # Chessboard Calibration for stereo pairs
 #
 # Paulo Dias (modified)

import numpy as np
import cv2
import glob
import os

# Board Size (internal corners)
board_h = 9
board_w = 6


def find_and_display_chessboard(img, show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)
    if ret:
        # refine corners to subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if show:
            img_draw = img.copy()
            cv2.drawChessboardCorners(img_draw, (board_w, board_h), corners, ret)
            cv2.imshow('img', img_draw)
            cv2.waitKey(500)
    return ret, corners


# prepare object points, like (0,0,0), (1,0,0), ...
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Lists to store points for all valid stereo pairs
objpoints = []           # 3d point in real world space (per stereo pair)
imgpoints_left = []      # 2d points in left image (per stereo pair)
imgpoints_right = []     # 2d points in right image (per stereo pair)

# Read left images and pair with corresponding right images
left_images = sorted(glob.glob('..//images//left*.jpg'))

for left_fname in left_images:
    right_fname = left_fname.replace('left', 'right')
    if not os.path.exists(right_fname):
        print('Right image not found for', left_fname)
        continue

    imgL = cv2.imread(left_fname)
    imgR = cv2.imread(right_fname)
    if imgL is None or imgR is None:
        print('Could not read pair:', left_fname, right_fname)
        continue

    retL, cornersL = find_and_display_chessboard(imgL, show=False)
    retR, cornersR = find_and_display_chessboard(imgR, show=False)

    if retL and retR:
        objpoints.append(objp.copy())
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
    else:
        print('Chessboard not found in pair:', left_fname, right_fname)

cv2.destroyAllWindows()

# Build the requested matrices: left_corners, right_corners and objPoints
if len(imgpoints_left) == 0:
    print('No valid stereo pairs with detected corners were found.')
    left_corners = np.empty((0, 2), dtype=np.float32)
    right_corners = np.empty((0, 2), dtype=np.float32)
    objPoints = np.empty((0, 3), dtype=np.float32)
else:
    left_corners = np.vstack([np.squeeze(p) for p in imgpoints_left]).astype(np.float32)
    right_corners = np.vstack([np.squeeze(p) for p in imgpoints_right]).astype(np.float32)
    objPoints = np.vstack([p for p in objpoints]).astype(np.float32)

    # Backwards-compatible alias for possible typo
    right_cornes = right_corners

    print('Found {} valid stereo pairs.'.format(len(imgpoints_left)))
    print('left_corners shape:', left_corners.shape)
    print('right_corners shape:', right_corners.shape)
    print('objPoints shape:', objPoints.shape)

    # Save matrices for later calibration steps
    np.savez('stereo_corners.npz', left_corners=left_corners, right_corners=right_corners, objPoints=objPoints)