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


def undistort_image(img, K, dist):
	h, w = img.shape[:2]
	newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
	und = cv2.undistort(img, K, dist, None, newK)
	if roi is not None:
		x, y, rw, rh = roi
		if rw > 0 and rh > 0:
			und = und[y:y + rh, x:x + rw]
	return und


def draw_epiline(img, line, color):
	a, b, c = line
	h, w = img.shape[:2]

	if abs(b) > 1e-9:
		y0 = int(round((-c - a * 0) / b))
		y1 = int(round((-c - a * (w - 1)) / b))
		pt1 = (0, y0)
		pt2 = (w - 1, y1)
	elif abs(a) > 1e-9:
		x = int(round(-c / a))
		pt1 = (x, 0)
		pt2 = (x, h - 1)
	else:
		return

	cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA)


def mouse_handler(event, x, y, flags, params):
	if event != cv2.EVENT_LBUTTONDOWN:
		return

	print('left click')
	print(f"{params['name']} point: ({x}, {y})")

	F = params['F']
	p = np.asarray([x, y], dtype=np.float32)
	color = np.random.randint(0, 255, 3).tolist()

	if params['which_image'] == 1:
		epiline = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), 1, F)
		epiline = epiline.reshape(-1, 3)[0]
		cv2.circle(params['self_img'], (x, y), 4, color, -1, cv2.LINE_AA)
		draw_epiline(params['other_img'], epiline, color)
	else:
		epiline = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), 2, F)
		epiline = epiline.reshape(-1, 3)[0]
		cv2.circle(params['self_img'], (x, y), 4, color, -1, cv2.LINE_AA)
		draw_epiline(params['other_img'], epiline, color)

	cv2.imshow(params['self_window'], params['self_img'])
	cv2.imshow(params['other_window'], params['other_img'])


def main():
	ap = argparse.ArgumentParser(description='Stereo epipolar lines on undistorted images')
	ap.add_argument('--left-pattern', '-p', default='..//images//left*.jpg', help='Glob pattern for left images')
	ap.add_argument('--index', '-i', type=int, default=0, help='Index of the left image to pick from pattern')
	ap.add_argument('--left', help='Explicit left image path (overrides pattern)')
	ap.add_argument('--right', help='Explicit right image path')
	args = ap.parse_args()

	if not os.path.exists('stereoParams.npz'):
		print('stereoParams.npz not found. Run ex8.2.py first.')
		sys.exit(1)

	data = np.load('stereoParams.npz')
	K1 = data['intrinsics1']
	dist1 = np.asarray(data['distortion1']).ravel()
	K2 = data['intrinsics2']
	dist2 = np.asarray(data['distortion2']).ravel()
	F = data['F']

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

	undL = undistort_image(imgL, K1, dist1)
	undR = undistort_image(imgR, K2, dist2)

	winL = 'Left undistorted'
	winR = 'Right undistorted'

	cv2.imshow(winL, undL)
	cv2.imshow(winR, undR)

	params_left = {
		'name': 'Left',
		'F': F,
		'which_image': 1,
		'self_img': undL,
		'other_img': undR,
		'self_window': winL,
		'other_window': winR,
	}
	params_right = {
		'name': 'Right',
		'F': F,
		'which_image': 2,
		'self_img': undR,
		'other_img': undL,
		'self_window': winR,
		'other_window': winL,
	}

	cv2.setMouseCallback(winL, mouse_handler, params_left)
	cv2.setMouseCallback(winR, mouse_handler, params_right)

	print('Left click on either image to mark a point and draw the corresponding epipolar line.')
	cv2.waitKey(-1)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
