"""9.4 - Iterative plane segmentation for the office point cloud scene."""

from copy import deepcopy
from pathlib import Path

import open3d as o3d


def load_point_cloud(file_path: Path, color=None):
	pcd = o3d.io.read_point_cloud(str(file_path))
	pcd.remove_non_finite_points()
	pcd = pcd.voxel_down_sample(voxel_size=0.05)
	if color is not None:
		pcd.paint_uniform_color(color)
	return pcd


def build_scene(script_dir: Path):
	merged_file = script_dir / "merged_offices.ply"
	if merged_file.exists():
		return load_point_cloud(merged_file)

	data_dir = script_dir.parent / "depth_images"
	source = load_point_cloud(data_dir / "office1.pcd", [0.9, 0.2, 0.2])
	target = load_point_cloud(data_dir / "office2.pcd", [0.2, 0.4, 0.9])
	return target + source


def segment_three_planes(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
	remaining = deepcopy(pcd)
	planes = []
	colors = [
		[0.9, 0.2, 0.2],
		[0.2, 0.8, 0.2],
		[0.2, 0.4, 0.9],
	]

	for plane_index in range(3):
		if len(remaining.points) < ransac_n:
			break

		plane_model, inliers = remaining.segment_plane(
			distance_threshold=distance_threshold,
			ransac_n=ransac_n,
			num_iterations=num_iterations,
		)
		plane_cloud = remaining.select_by_index(inliers)
		plane_cloud.paint_uniform_color(colors[plane_index % len(colors)])
		planes.append((plane_model, plane_cloud))
		remaining = remaining.select_by_index(inliers, invert=True)

	return planes, remaining


def main():
	script_dir = Path(__file__).resolve().parent
	scene = build_scene(script_dir)

	if scene.is_empty():
		raise RuntimeError("No point cloud data available for plane segmentation.")

	planes, remaining = segment_three_planes(scene)
	axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

	for index, (plane_model, plane_cloud) in enumerate(planes, start=1):
		print(f"Plane {index}: {plane_model}")
		print(f"Plane {index} points: {len(plane_cloud.points)}")

	if not remaining.is_empty():
		remaining.paint_uniform_color([0.7, 0.7, 0.7])
		print("Remaining points:", len(remaining.points))

	geometries = [plane_cloud for _, plane_cloud in planes]
	if not remaining.is_empty():
		geometries.append(remaining)
	geometries.append(axes)
	o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
	main()