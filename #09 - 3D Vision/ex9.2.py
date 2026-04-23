# viewcloud.py
#
# open3D example to view Kinect PCD point clouds
#
# Paulo Dias

from pathlib import Path

import open3d as o3d


def load_point_cloud(file_path: Path, color):
	pcd = o3d.io.read_point_cloud(str(file_path))
	pcd.remove_non_finite_points()
	if not pcd.is_empty():
		pcd = pcd.voxel_down_sample(voxel_size=0.05)
		pcd.paint_uniform_color(color)
	return pcd


base_dir = Path(__file__).resolve().parents[1] / "depth_images"

office1 = load_point_cloud(base_dir / "office1.pcd", [0.9, 0.2, 0.2])
office2 = load_point_cloud(base_dir / "office2.pcd", [0.2, 0.4, 0.9])

# Create axes mesh
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# Show point clouds in view
o3d.visualization.draw_geometries([office1, office2, axes])