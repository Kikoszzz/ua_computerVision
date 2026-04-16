import numpy as np
import cv2
import glob
import os
import argparse
import sys

# Stereo calibration using OpenCV's stereoCalibrate

# Board Size (internal corners)
board_w = 6
board_h = 9
corners_per_view = board_w * board_h


def load_stereo_corners(npz_path):
	data = np.load(npz_path)
	left = data['left_corners']
	right = data['right_corners']
	obj = data['objPoints']
	return left, right, obj


def stacked_to_lists(left, right, obj):
	if left.shape[0] % corners_per_view != 0:
		raise ValueError('Left corners count not divisible by corners per view')
	nviews = left.shape[0] // corners_per_view
	obj_list = [obj[i*corners_per_view:(i+1)*corners_per_view].astype(np.float32) for i in range(nviews)]
	left_list = [left[i*corners_per_view:(i+1)*corners_per_view].astype(np.float32).reshape(-1,1,2) for i in range(nviews)]
	right_list = [right[i*corners_per_view:(i+1)*corners_per_view].astype(np.float32).reshape(-1,1,2) for i in range(nviews)]
	return obj_list, left_list, right_list


def detect_corners_from_pairs(left_pattern, show=False):
	left_files = sorted(glob.glob(left_pattern))
	if len(left_files) == 0:
		raise FileNotFoundError('No left images found for pattern: ' + left_pattern)

	objp = np.zeros((corners_per_view, 3), np.float32)
	objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

	objs = []
	imgptsL = []
	imgptsR = []
	imageSize = None

	for lf in left_files:
		rf = lf.replace('left', 'right')
		if not os.path.exists(rf):
			print('Skipping, right image not found for', lf)
			continue
		imgL = cv2.imread(lf)
		imgR = cv2.imread(rf)
		if imgL is None or imgR is None:
			print('Skipping unreadable pair:', lf, rf)
			continue
		if imageSize is None:
			imageSize = (imgL.shape[1], imgL.shape[0])

		grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
		grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

		retL, cornersL = cv2.findChessboardCorners(grayL, (board_w, board_h), None)
		retR, cornersR = cv2.findChessboardCorners(grayR, (board_w, board_h), None)

		if retL and retR:
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
			cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
			cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
			objs.append(objp.copy())
			imgptsL.append(cornersL.reshape(-1,1,2).astype(np.float32))
			imgptsR.append(cornersR.reshape(-1,1,2).astype(np.float32))
			if show:
				cv2.drawChessboardCorners(imgL, (board_w, board_h), cornersL, retL)
				cv2.drawChessboardCorners(imgR, (board_w, board_h), cornersR, retR)
				cv2.imshow('left', imgL)
				cv2.imshow('right', imgR)
				cv2.waitKey(500)
		else:
			print('Chessboard not found in pair:', lf, rf)

	if show:
		cv2.destroyAllWindows()

	return objs, imgptsL, imgptsR, imageSize


def main():
	parser = argparse.ArgumentParser(description='Stereo calibration using chessboard pairs')
	parser.add_argument('--pattern', '-p', default='..//images//left*.jpg', help='Glob pattern for left images')
	parser.add_argument('--force-detect', action='store_true', help='Force corner detection (ignore existing stereo_corners.npz)')
	parser.add_argument('--show', action='store_true', help='Show detected corners')
	args = parser.parse_args()

	# Try to load previously saved corner stacks
	if os.path.exists('stereo_corners.npz') and not args.force_detect:
		print('Loading stacked corner arrays from stereo_corners.npz')
		left_c, right_c, obj_c = load_stereo_corners('stereo_corners.npz')
		objs, imgL_list, imgR_list = stacked_to_lists(left_c, right_c, obj_c)
		# determine image size from first matching left image
		files = sorted(glob.glob(args.pattern))
		if len(files) == 0:
			raise FileNotFoundError('No left images found to determine image size')
		tmp = cv2.imread(files[0])
		imageSize = (tmp.shape[1], tmp.shape[0])
	else:
		print('Detecting corners from image pairs using pattern:', args.pattern)
		objs, imgL_list, imgR_list, imageSize = detect_corners_from_pairs(args.pattern, show=args.show)

	if len(objs) == 0:
		print('No valid stereo views with detected corners. Exiting.')
		sys.exit(1)

	print('Number of valid views:', len(objs))
	print('Image size:', imageSize)

	# stereoCalibrate expects lists of arrays
	flags = cv2.CALIB_SAME_FOCAL_LENGTH
	criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

	try:
		ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
			objs, imgL_list, imgR_list, None, None, None, None, imageSize,
			flags=flags, criteria=criteria)
	except Exception as e:
		print('stereoCalibrate failed:', e)
		sys.exit(1)

	print('stereoCalibrate RMS error:', ret)
	print('intrinsics1 shape:', cameraMatrix1.shape)
	print('distortion1 shape:', distCoeffs1.shape)
	print('intrinsics2 shape:', cameraMatrix2.shape)
	print('distortion2 shape:', distCoeffs2.shape)
	print('R shape:', R.shape)
	print('T shape:', T.shape)

	np.savez('stereoParams.npz',
			 intrinsics1=cameraMatrix1,
			 distortion1=distCoeffs1,
			 intrinsics2=cameraMatrix2,
			 distortion2=distCoeffs2,
			 R=R, T=T, E=E, F=F)

	print('Saved stereo parameters to stereoParams.npz')


if __name__ == '__main__':
	main()

