import os
import re
import glob
import sys
import argparse

import numpy as np
import cv2


def find_right_for_left(left_path):
	cand = left_path.replace('left', 'right', 1)
	if os.path.exists(cand):
		return cand

	base = os.path.basename(left_path)
	d = os.path.dirname(left_path)
	m = re.search(r"(\d+)", base)
	if m:
		num = m.group(1)
		candidates = sorted(glob.glob(os.path.join(d, f"*{num}.*")))
		for c in candidates:
			if 'right' in os.path.basename(c).lower():
				return c

	rights = sorted(glob.glob(os.path.join(d, 'right*.*')))
	if rights:
		return rights[0]

	return None


def load_pair(args):
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

	if imgL.shape[:2] != imgR.shape[:2]:
		print('Left and right images must have the same size for rectification')
		sys.exit(1)

	return imgL, imgR


def rectify_pair(imgL, imgR, intrinsics1, distortion1, intrinsics2, distortion2, R, T):
	height, width = imgL.shape[:2]

	R1 = np.zeros(shape=(3, 3), dtype=np.float64)
	R2 = np.zeros(shape=(3, 3), dtype=np.float64)
	P1 = np.zeros(shape=(3, 4), dtype=np.float64)
	P2 = np.zeros(shape=(3, 4), dtype=np.float64)
	Q = np.zeros(shape=(4, 4), dtype=np.float64)

	R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
		intrinsics1,
		distortion1,
		intrinsics2,
		distortion2,
		(width, height),
		R,
		T,
		flags=cv2.CALIB_ZERO_DISPARITY,
		alpha=-1,
		newImageSize=(0, 0),
	)

	print('InitUndistortRectifyMap')
	map1x, map1y = cv2.initUndistortRectifyMap(
		intrinsics1,
		distortion1,
		R1,
		P1,
		(width, height),
		cv2.CV_32FC1,
	)
	map2x, map2y = cv2.initUndistortRectifyMap(
		intrinsics2,
		distortion2,
		R2,
		P2,
		(width, height),
		cv2.CV_32FC1,
	)

	rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
	rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
	return rectL, rectR, Q


def main():
	ap = argparse.ArgumentParser(description='Disparity map using StereoBM on rectified images')
	ap.add_argument('--left-pattern', '-p', default='..//images//left*.jpg', help='Glob pattern for left images')
	ap.add_argument('--index', '-i', type=int, default=0, help='Index of the left image to pick from pattern')
	ap.add_argument('--left', help='Explicit left image path (overrides pattern)')
	ap.add_argument('--right', help='Explicit right image path')
	args = ap.parse_args()

	if not os.path.exists('stereoParams.npz'):
		print('stereoParams.npz not found. Run ex8.2.py first.')
		sys.exit(1)

	data = np.load('stereoParams.npz')
	intrinsics1 = data['intrinsics1']
	distortion1 = np.asarray(data['distortion1']).ravel()
	intrinsics2 = data['intrinsics2']
	distortion2 = np.asarray(data['distortion2']).ravel()
	R = data['R']
	T = data['T']

	left, right = load_pair(args)
	remap_imgl, remap_imgr, Q = rectify_pair(left, right, intrinsics1, distortion1, intrinsics2, distortion2, R, T)

	# StereoBM requires grayscale rectified images.
	grayL = cv2.cvtColor(remap_imgl, cv2.COLOR_BGR2GRAY)
	grayR = cv2.cvtColor(remap_imgr, cv2.COLOR_BGR2GRAY)

	# Call the constructor for StereoBM
	stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=21)

	# Calculate the disparity image
	disparity = stereo.compute(grayL, grayR)

	# Display as a CV_8UC1 image
	disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
	disparity = np.uint8(disparity)

	cv2.imshow('left', remap_imgl)
	cv2.imshow('Disparity Map', disparity)
	print('Press any key to exit.')
	cv2.waitKey(-1)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
