import os
import re
import glob
import sys
import argparse

import numpy as np
import cv2


def find_right_for_left(left_path):
    # quick replace
    cand = left_path.replace('left', 'right', 1)
    if os.path.exists(cand):
        return cand

    # try to find same numeric suffix
    base = os.path.basename(left_path)
    d = os.path.dirname(left_path)
    m = re.search(r"(\d+)", base)
    if m:
        num = m.group(1)
        candidates = sorted(glob.glob(os.path.join(d, f"*{num}.*")))
        for c in candidates:
            if 'right' in os.path.basename(c).lower():
                return c

    # fallback: any file starting with 'right'
    rights = sorted(glob.glob(os.path.join(d, 'right*.*')))
    if rights:
        return rights[0]

    return None


def undistort_image(img, K, dist):
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    und = cv2.undistort(img, K, dist, None, newK)
    if roi is not None:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            und = und[y:y + rh, x:x + rw]
    return und, newK, roi


def main():
    ap = argparse.ArgumentParser(description='Undistort a stereo pair using stereoParams.npz')
    ap.add_argument('--left-pattern', '-p', default='..//images//left*.jpg', help='Glob pattern for left images')
    ap.add_argument('--index', '-i', type=int, default=0, help='Index of the left image to pick from pattern')
    ap.add_argument('--left', help='Explicit left image path (overrides pattern)')
    ap.add_argument('--right', help='Explicit right image path')
    ap.add_argument('--save', action='store_true', help='Save undistorted images to disk')
    ap.add_argument('--outdir', default='.', help='Output directory for saved images')
    args = ap.parse_args()

    if not os.path.exists('stereoParams.npz'):
        print('stereoParams.npz not found. Run the calibration (ex8.2.py) first.')
        sys.exit(1)

    data = np.load('stereoParams.npz')
    K1 = data['intrinsics1']
    dist1 = np.asarray(data['distortion1']).ravel()
    K2 = data['intrinsics2']
    dist2 = np.asarray(data['distortion2']).ravel()

    print('Loaded stereoParams.npz')
    print('intrinsics1', K1.shape, 'distortion1', dist1.shape)
    print('intrinsics2', K2.shape, 'distortion2', dist2.shape)

    if args.left:
        left_path = args.left
    else:
        left_files = sorted(glob.glob(args.left_pattern))
        if len(left_files) == 0:
            print('No left images found for pattern:', args.left_pattern)
            sys.exit(1)
        idx = args.index
        if idx < 0 or idx >= len(left_files):
            idx = 0
        left_path = left_files[idx]

    if args.right:
        right_path = args.right
    else:
        right_path = find_right_for_left(left_path)
        if right_path is None:
            print('Could not find matching right image for', left_path)
            sys.exit(1)

    print('Using pair:')
    print('  left :', left_path)
    print('  right:', right_path)

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    if imgL is None or imgR is None:
        print('Failed to read images')
        sys.exit(1)

    undL, newK1, roi1 = undistort_image(imgL, K1, dist1)
    undR, newK2, roi2 = undistort_image(imgR, K2, dist2)

    # show images
    cv2.imshow('Left original', imgL)
    cv2.imshow('Left undistorted', undL)
    cv2.imshow('Right original', imgR)
    cv2.imshow('Right undistorted', undR)
    print('Press s to save undistorted images (or use --save), any other key to exit')
    k = cv2.waitKey(0) & 0xFF

    if k == ord('s') or args.save:
        os.makedirs(args.outdir, exist_ok=True)
        left_out = os.path.join(args.outdir, os.path.basename(left_path).rsplit('.', 1)[0] + '_und.jpg')
        right_out = os.path.join(args.outdir, os.path.basename(right_path).rsplit('.', 1)[0] + '_und.jpg')
        cv2.imwrite(left_out, undL)
        cv2.imwrite(right_out, undR)
        print('Saved:', left_out)
        print('Saved:', right_out)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
