import os
import sys
import cv2
import numpy as np

"""
ex7.5.py

ArUco marker pose estimation using calibrated camera parameters.

Features:
- Loads camera intrinsics/distortion from NPZ (camera_7.3.npz, camera.npz, camera_params.npz).
- Detects ArUco markers in live camera feed.
- Estimates pose for each marker with solvePnP.
- Draws marker borders and 3D axes.
- Optional: saves a generated marker image for printing/testing.

Keys:
- q: quit
- g: generate and save marker image (marker_<id>.png)
"""

# Detection/configuration parameters
CAMERA_ID = 0
MARKER_LENGTH_M = 0.05  # Real marker side length in meters
AXIS_LENGTH_M = MARKER_LENGTH_M * 0.7
DICTIONARY_ID = cv2.aruco.DICT_6X6_250
MARKER_ID_TO_GENERATE = 23
MARKER_IMAGE_SIZE = 400


def ensure_aruco_available() -> None:
    if not hasattr(cv2, "aruco"):
        print("Error: cv2.aruco is not available in this OpenCV build.")
        print("Install opencv-contrib-python (matching your OpenCV version).")
        sys.exit(1)


def load_camera_params():
    candidates = ["camera_7.3.npz", "camera.npz", "camera_params.npz"]
    for path in candidates:
        if os.path.exists(path):
            with np.load(path) as data:
                intrinsics = data["intrinsics"]
                distortion = data["distortion"]
            print(f"Loaded camera parameters from: {path}")
            print("Intrinsics:\n", intrinsics)
            print("Distortion:\n", distortion)
            return intrinsics, distortion

    print("No camera parameter file found.")
    print("Expected one of: camera_7.3.npz, camera.npz, camera_params.npz")
    print("Run calibration first (ex7.3.py or chessboard.py).")
    sys.exit(1)


def create_marker_image(dictionary):
    marker = cv2.aruco.generateImageMarker(
        dictionary,
        MARKER_ID_TO_GENERATE,
        MARKER_IMAGE_SIZE,
    )
    out_name = f"marker_{MARKER_ID_TO_GENERATE}.png"
    cv2.imwrite(out_name, marker)
    print(f"Saved marker image: {out_name}")


def marker_object_points(marker_length_m: float) -> np.ndarray:
    half = marker_length_m / 2.0
    # Corner order must match ArUco detectMarkers output:
    # top-left, top-right, bottom-right, bottom-left
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def detect_and_draw_pose(frame, detector, obj_pts, intrinsics, distortion, print_pose=False):
    corners, ids, rejected = detector.detectMarkers(frame)
    out = frame.copy()
    poses = []

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i].reshape(-1, 2).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                image_points,
                intrinsics,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success:
                cv2.drawFrameAxes(
                    out,
                    intrinsics,
                    distortion,
                    rvec,
                    tvec,
                    AXIS_LENGTH_M,
                    2,
                )
                distance_m = float(np.linalg.norm(tvec))
                origin = tuple(image_points[0].astype(int))
                cv2.putText(
                    out,
                    f"id={marker_id} z={tvec[2, 0]:.3f}m d={distance_m:.3f}m",
                    (origin[0], max(20, origin[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                poses.append((int(marker_id), rvec.ravel(), tvec.ravel(), distance_m))

    if rejected is not None and len(rejected) > 0:
        cv2.aruco.drawDetectedMarkers(out, rejected, borderColor=(100, 0, 255))

    if print_pose:
        if len(poses) == 0:
            print("No valid marker pose estimated in this frame/image.")
        for marker_id, rvec, tvec, distance_m in poses:
            print(f"id={marker_id} rvec={rvec} tvec={tvec} d={distance_m:.3f}m")

    return out, len(poses)


def run_camera_mode(detector, obj_pts, intrinsics, distortion):
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Could not open camera with id={CAMERA_ID}")
        return

    print("Running camera mode. Press 'q' to quit, 'g' to save marker image.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera")
            break

        out, _ = detect_and_draw_pose(frame, detector, obj_pts, intrinsics, distortion, print_pose=False)
        cv2.imshow("Aruco Pose Estimation - Camera", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            create_marker_image(detector.getDictionary())

    cap.release()
    cv2.destroyAllWindows()


def run_image_mode(detector, obj_pts, intrinsics, distortion):
    print("Image mode selected.")
    print("Example path: aruco_photo.jpg")
    image_path = input("Image path: ").strip().strip('"')
    if not image_path:
        print("No image path provided.")
        return
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    out, n_poses = detect_and_draw_pose(image, detector, obj_pts, intrinsics, distortion, print_pose=True)
    print(f"Detected poses: {n_poses}")

    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_pose{ext if ext else '.png'}"
    cv2.imwrite(out_path, out)
    print(f"Saved output image: {out_path}")

    cv2.imshow("Aruco Pose Estimation - Image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_menu():
    print("\n=== ArUco Pose Estimation Menu ===")
    print("1) Camera mode")
    print("2) Image mode")
    print("3) Generate marker image")
    print("0) Exit")
    return input("Choose an option: ").strip()


def main():
    ensure_aruco_available()
    intrinsics, distortion = load_camera_params()

    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    obj_pts = marker_object_points(MARKER_LENGTH_M)

    while True:
        option = show_menu()
        if option == "1":
            run_camera_mode(detector, obj_pts, intrinsics, distortion)
        elif option == "2":
            run_image_mode(detector, obj_pts, intrinsics, distortion)
        elif option == "3":
            create_marker_image(dictionary)
        elif option == "0":
            print("Exiting.")
            break
        else:
            print("Invalid option. Choose 1, 2, 3 or 0.")


if __name__ == "__main__":
    main()
