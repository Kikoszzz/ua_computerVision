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


def draw_horizontal_grid(img, step=25, color=(0, 255, 0), thickness=1):
	h, w = img.shape[:2]
	for y in range(0, h, step):
		cv2.line(img, (0, y), (w - 1, y), color, thickness, cv2.LINE_AA)


def row_mouse_handler(event, x, y, flags, params):
	if event != cv2.EVENT_LBUTTONDOWN:
		return

	print('left click')
	print(f"{params['name']} point: ({x}, {y})")

	self_img = params['self_img']
	other_img = params['other_img']
	self_clean = params['self_clean']
	other_clean = params['other_clean']

	self_img[:] = self_clean
	other_img[:] = other_clean

	# Mark selected point in current image and corresponding row in the other image.
	color = np.random.randint(0, 255, 3).tolist()
	cv2.circle(self_img, (x, y), 4, color, -1, cv2.LINE_AA)
	cv2.line(other_img, (0, y), (other_img.shape[1] - 1, y), color, 2, cv2.LINE_AA)

	cv2.imshow(params['self_window'], self_img)
	cv2.imshow(params['other_window'], other_img)


def main():
	ap = argparse.ArgumentParser(description='Stereo image rectification and row alignment visualization')
	ap.add_argument('--left-pattern', '-p', default='..//images//left*.jpg', help='Glob pattern for left images')
	ap.add_argument('--index', '-i', type=int, default=0, help='Index of the left image to pick from pattern')
	ap.add_argument('--left', help='Explicit left image path (overrides pattern)')
	ap.add_argument('--right', help='Explicit right image path')
	ap.add_argument('--step', type=int, default=25, help='Row spacing for horizontal guide lines')
	ap.add_argument('--interactive', action='store_true', help='Enable click interaction to show corresponding row')
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

	rectL_grid = rectL.copy()
	rectR_grid = rectR.copy()
	draw_horizontal_grid(rectL_grid, step=max(1, args.step))
	draw_horizontal_grid(rectR_grid, step=max(1, args.step))

	winL = 'Left rectified'
	winR = 'Right rectified'
	cv2.imshow(winL, rectL_grid)
	cv2.imshow(winR, rectR_grid)

	if args.interactive:
		print('Interactive mode: click a point to show corresponding row in the other image.')
		params_left = {
			'name': 'Left',
			'self_img': rectL_grid,
			'other_img': rectR_grid,
			'self_clean': rectL.copy(),
			'other_clean': rectR.copy(),
			'self_window': winL,
			'other_window': winR,
		}
		params_right = {
			'name': 'Right',
			'self_img': rectR_grid,
			'other_img': rectL_grid,
			'self_clean': rectR.copy(),
			'other_clean': rectL.copy(),
			'self_window': winR,
			'other_window': winL,
		}

		cv2.setMouseCallback(winL, row_mouse_handler, params_left)
		cv2.setMouseCallback(winR, row_mouse_handler, params_right)

	print('Press any key to exit.')
	cv2.waitKey(-1)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
