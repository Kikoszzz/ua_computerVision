"""9.3.1 - ICP alignment for the office Kinect point clouds.

This version supports the default automatic ICP alignment and an optional
manual initialization step based on point picking.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(file_path: Path, color):
	pcd = o3d.io.read_point_cloud(str(file_path))
	pcd.remove_non_finite_points()
	pcd = pcd.voxel_down_sample(voxel_size=0.05)
	pcd.paint_uniform_color(color)
	return pcd


def pick_points(pcd):
	print("")
	print("1) Please pick at least three correspondences using [shift + left click]")
	print("   Press [shift + right click] to undo point picking")
	print("2) After picking points, press q to close the window")
	vis = o3d.visualization.VisualizerWithEditing()
	vis.create_window()
	vis.add_geometry(pcd)
	vis.run()
	vis.destroy_window()
	print("")
	return vis.get_picked_points()


def compute_initial_transform(source, target):
	use_manual_correspondences = False
	if not use_manual_correspondences:
		return np.eye(4)

	picked_id_source = pick_points(source)
	picked_id_target = pick_points(target)
	assert len(picked_id_source) >= 3 and len(picked_id_target) >= 3
	assert len(picked_id_source) == len(picked_id_target)

	corr = np.zeros((len(picked_id_source), 2), dtype=np.int32)
	corr[:, 0] = picked_id_source
	corr[:, 1] = picked_id_target

	correspondences = o3d.utility.Vector2iVector(corr)
	estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
	return estimation.compute_transformation(source, target, correspondences)


def main():
	script_dir = Path(__file__).resolve().parent
	data_dir = script_dir.parent / "depth_images"

	source = load_point_cloud(data_dir / "office1.pcd", [0.9, 0.2, 0.2])
	target = load_point_cloud(data_dir / "office2.pcd", [0.2, 0.4, 0.9])

	if source.is_empty() or target.is_empty():
		raise RuntimeError("Failed to load one or both point clouds.")

	print("Source points:", len(source.points))
	print("Target points:", len(target.points))

	threshold = 0.05
	initial_transform = compute_initial_transform(source, target)
	result = o3d.pipelines.registration.registration_icp(
		source,
		target,
		threshold,
		initial_transform,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
	)

	print("ICP fitness:", result.fitness)
	print("ICP inlier RMSE:", result.inlier_rmse)
	print("ICP transformation:\n", result.transformation)

	original_source = deepcopy(source)
	original_source.paint_uniform_color([0.7, 0.1, 0.1])
	target_view = deepcopy(target)
	target_view.paint_uniform_color([0.1, 0.2, 0.7])
	aligned_source = deepcopy(source)
	aligned_source.transform(result.transformation)
	aligned_source.paint_uniform_color([0.95, 0.65, 0.1])

	axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
	o3d.visualization.draw_geometries([original_source, target_view, aligned_source, axes])

	merged = target + aligned_source
	output_file = script_dir / "merged_offices.ply"
	o3d.io.write_point_cloud(str(output_file), merged)
	print("Saved merged point cloud to", output_file)

	merged_view = deepcopy(merged)
	merged_view.paint_uniform_color([0.3, 0.7, 0.3])
	o3d.visualization.draw_geometries([merged_view, axes])


if __name__ == "__main__":
	main()