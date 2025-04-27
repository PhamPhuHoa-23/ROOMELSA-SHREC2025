import os
import json
import numpy as np
import open3d as o3d
from simpleicp import PointCloud, SimpleICP
import copy

def normalize_point_cloud(pcd):
    """
    Normalize point cloud to be centered at origin and scaled to unit sphere
    """
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(centered_points ** 2, axis=1)))
    normalized_points = centered_points / max_dist

    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)

    if pcd.has_colors():
        normalized_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    if pcd.has_normals():
        normalized_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals))

    return normalized_pcd, centroid, max_dist

def load_point_cloud(file_path, is_npy=False, is_src=False):
    """
    Load point cloud from file
    """
    if is_npy:
        data = np.load(file_path)
        pcd = o3d.geometry.PointCloud()
        points = data[:, :3]
        if is_src: points[:, [1, 2]] = points[:, [2, 1]]
        pcd.points = o3d.utility.Vector3dVector(points)
        if data.shape[1] >= 6:
            colors = data[:, 3:6]
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd = o3d.io.read_point_cloud(file_path)
    return pcd


def visualize_registration(source, target, transformed_source=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0, 0])  # Red
    target_temp.paint_uniform_color([0, 1, 0])  # Green

    if transformed_source is None:
        o3d.visualization.draw_geometries([target_temp])
    else:
        transformed_source_temp = copy.deepcopy(transformed_source)
        transformed_source_temp.paint_uniform_color([0, 0, 1])  # Blue
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        o3d.visualization.draw_geometries([target_temp])

def pca_alignment(source, target):
    """
    Align source point cloud to target using PCA
    """
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center both point clouds
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute covariance matrices
    source_cov = np.cov(source_centered.T)
    target_cov = np.cov(target_centered.T)

    # Compute eigenvectors and eigenvalues
    source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
    target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)

    # Sort eigenvectors by eigenvalues in descending order
    source_idx = np.argsort(source_eigvals)[::-1]
    target_idx = np.argsort(target_eigvals)[::-1]

    source_eigvecs = source_eigvecs[:, source_idx]
    target_eigvecs = target_eigvecs[:, target_idx]

    # Compute rotation matrix from source to target
    rotation = np.dot(target_eigvecs, source_eigvecs.T)

    # Ensure it's a proper rotation matrix (det = 1)
    # if np.linalg.det(rotation) < 0:
    #     source_eigvecs[:, 2] = -source_eigvecs[:, 2]
    #     rotation = np.dot(target_eigvecs, source_eigvecs.T)

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation

    # Add translation to move source centroid to target centroid after rotation
    transformation[:3, 3] = target_centroid - np.dot(rotation, source_centroid)

    # Apply transformation to source
    aligned_source = copy.deepcopy(source)
    aligned_source.transform(transformation)

    return aligned_source, transformation


def chamfer_distance(source, target):
    """
    Tính Chamfer Distance giữa hai điểm mây.

    Parameters:
    -----------
    source : o3d.geometry.PointCloud
        Điểm mây nguồn
    target : o3d.geometry.PointCloud
        Điểm mây đích

    Returns:
    --------
    float
        Giá trị Chamfer Distance giữa hai điểm mây
    """
    # Chuyển đổi điểm mây sang mảng numpy
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # Xây dựng KDTree cho tập điểm đích để tìm kiếm nhanh
    target_tree = o3d.geometry.KDTreeFlann(target)

    # Tính tổng khoảng cách từ source đến target
    source_to_target = 0.0
    for point in source_points:
        # Tìm điểm gần nhất trong target cho mỗi điểm trong source
        _, idx, dist = target_tree.search_knn_vector_3d(point, 1)
        source_to_target += np.sqrt(dist[0])

    # Xây dựng KDTree cho tập điểm nguồn
    source_tree = o3d.geometry.KDTreeFlann(source)

    # Tính tổng khoảng cách từ target đến source
    target_to_source = 0.0
    for point in target_points:
        # Tìm điểm gần nhất trong source cho mỗi điểm trong target
        _, idx, dist = source_tree.search_knn_vector_3d(point, 1)
        target_to_source += np.sqrt(dist[0])

    # Lấy trung bình theo số điểm trong mỗi tập
    source_to_target /= len(source_points)
    target_to_source /= len(target_points)
    target_to_source = 0

    # Chamfer distance là tổng của hai khoảng cách trung bình
    chamfer_dist = source_to_target + target_to_source

    return chamfer_dist

if __name__ == '__main__':
    # source_file = "D:\\private\\objects_dataset_npy_10000\\objects\\ee127828-041a-483e-836b-662b26b9cadb\\normalized_model.npy"
    source_file = "D:\\private\\objects_dataset_npy_10000\\objects\\a63e6333-b3b8-4487-b3ae-7c8c5e3092e8\\normalized_model.npy"
    target_file = "D:\\private\\scenes\\08df38e7-b9ec-40d1-8652-b1857959a6c7\\masked_cloud.ply"

    source_pcd = load_point_cloud(source_file, is_npy=True, is_src=True)
    target_pcd = load_point_cloud(target_file, is_npy=False)

    source_pcd, _, _ = normalize_point_cloud(source_pcd)
    target_pcd, _, _ = normalize_point_cloud(target_pcd)

    visualize_registration(source_pcd, target_pcd)

    source_pcd, _ = pca_alignment(source_pcd, target_pcd)

    source_npy = source_pcd.points
    target_npy = target_pcd.points


    # Create point cloud objects
    source_pcdd = PointCloud(source_npy, columns=["x", "y", "z"])
    target_pcdd = PointCloud(target_npy, columns=["x", "y", "z"])

    icp = SimpleICP()
    icp.add_point_clouds(target_pcdd, source_pcdd)
    H, source_pcd_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)

    print(source_pcd_transformed.shape)

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(source_pcd_transformed)

    visualize_registration(source_pcd, target_pcd, transformed_pcd)

    chamfer_dist = chamfer_distance(transformed_pcd, target_pcd)
    print(f"Chamfer Distance: {chamfer_dist}")

    # target_dir = "D:\\private\\scenes"
    # source_dir = "D:\\private\\objects_dataset_npy_10000\\objects\\"
    #
    # output_dict = {}
    #
    # for target_file in os.listdir(target_dir):
    #     target_path = os.path.join(target_dir, target_file)
    #     target_point_cloud_path = os.path.join(target_path, "masked_cloud.ply")
    #     target_pcd = load_point_cloud(target_point_cloud_path)
    #     target_pcd, _, _ = normalize_point_cloud(target_pcd)
    #     target_pcd_npy = target_pcd.points
    #     target_pcd_icp = PointCloud(target_pcd_npy, columns=["x", "y", "z"])
    #     output_dict[target_file] = {}
    #
    #     for source_file in os.listdir(source_dir):
    #         source_path = os.path.join(source_dir, source_file)
    #         source_point_cloud_path = os.path.join(source_path, "normalized_model.npy")
    #         source_pcd = load_point_cloud(source_point_cloud_path, is_npy=True, is_src=True)
    #         source_pcd, _, _ = normalize_point_cloud(source_pcd)
    #
    #         source_pcd_aligned, _ = pca_alignment(source_pcd, target_pcd)
    #         source_pcd_aligned_npy = source_pcd_aligned.points
    #         source_pcd_aligned_icp = PointCloud(source_pcd_aligned_npy, columns=["x", "y", "z"])
    #
    #         icp = SimpleICP()
    #         icp.add_point_clouds(target_pcd_icp, source_pcd_aligned_icp)
    #
    #         _, source_transformed, _, _ = icp.run(max_overlap_distance=1)
    #
    #         transformed_pcd = o3d.geometry.PointCloud()
    #         transformed_pcd.points = o3d.utility.Vector3dVector(source_transformed)
    #
    #         chamfer_dist = chamfer_distance(transformed_pcd, target_pcd)
    #
    #         output_dict[target_file][source_file] = chamfer_dist
    #
    #         print(target_file, source_file, chamfer_dist)
    #
    # with open("../RetrievalSystem/distance_mapping.json", "w", encoding="utf-8") as f:
    #     json.dump(output_dict, f, ensure_ascii=False, indent=4)














