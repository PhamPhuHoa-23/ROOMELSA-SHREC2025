import numpy as np
import open3d as o3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
import copy

from dgl.function import copy_u
from scipy.spatial import cKDTree


def load_point_cloud(file_path, is_npy=False):
    """
    Tải đám mây điểm từ file
    """
    if is_npy:
        data = np.load(file_path)
        pcd = o3d.geometry.PointCloud()

        # Lấy tọa độ xyz
        points = data[:, :3]
        pcd.points = o3d.utility.Vector3dVector(points)

        # Nếu có màu sắc
        if data.shape[1] >= 6:
            # Giả sử màu sắc đã được chuẩn hóa trong khoảng [0, 1]
            colors = data[:, 3:]
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Sử dụng cho các định dạng khác như .ply, .pcd, .obj
        pcd = o3d.io.read_point_cloud(file_path)
        # mesh = load_objs_as_meshes(file_path)
        # pcd = sample_points_from_meshes(mesh, num_samples=30000)

    return pcd

def normalize_point_cloud(pcd):
    points = np.asarray(pcd.points)

    centroid = np.mean(pcd.points, axis=0)
    centered_points = points - centroid

    max_dist = np.max(np.sqrt(np.sum(centered_points ** 2, axis=1)))
    normalized_points = centered_points / max_dist

    normalized_pcd = o3d.geometry.PointCloud()
    normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)

    return normalized_pcd, centroid, max_dist

def chamfer_distance(src_points, tar_points):
    src_tree = cKDTree(src_points)
    tar_tree = cKDTree(tar_points)

    src_to_tar_dist, _ = src_tree.query(tar_points)
    tar_to_src_dist, _ = tar_tree.query(src_points)

    chamfer_dist = np.mean(src_to_tar_dist) + np.mean(tar_to_src_dist)
    return chamfer_dist

def icp_with_normal_init(src_pcd, tar_pcd, max_iter=100, threshold=0.05):
    src_points = np.asarray(src_pcd.points)
    tar_points = np.asarray(tar_pcd.points)

    src_centroid = np.mean(src_points, axis=0)
    tar_centroid = np.mean(tar_points, axis=0)

    src_centered = src_points - src_centroid
    tar_centered = tar_points - tar_centroid

    src_cov = np.cov(src_centered.T)
    tar_cov = np.cov(tar_centered.T)

    src_eigvals, src_eigvecs = np.linalg.eig(src_cov)
    tar_eigvals, tar_eigvecs = np.linalg.eig(tar_cov)

    src_idx = np.argsort(src_eigvals)[::-1]
    tar_idx = np.argsort(tar_eigvals)[::-1]
    src_eigvecs = src_eigvecs[:, src_idx]
    tar_eigvecs = tar_eigvecs[:, tar_idx]

    R_init = np.dot(tar_eigvecs, src_eigvecs.T)
    if np.linalg.det(R_init) < 0:
        src_eigvecs[:,2] = -src_eigvecs[:,2]
        R_init = np.dot(tar_eigvecs, src_eigvecs.T)

    src_pcd_init = copy.deepcopy(src_pcd)
    init_transform = np.eye(4)
    init_transform[:3, :3] = R_init
    src_pcd_init.transform(init_transform)

    if tar_pcd.has_normals() and src_pcd.has_normals():
        # Sử dụng point-to-plane nếu có normal
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        # Sử dụng point-to-point nếu không có normal
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    # estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)

    icp_result = o3d.pipelines.registration.registration_icp(
        src_pcd_init,
        tar_pcd,
        threshold,
        np.eye(4),
        # estimation,
        # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )

    final_transform = np.dot(icp_result.transformation, init_transform)

    src_pcd_transformed = copy.deepcopy(src_pcd)
    src_pcd_transformed.transform(final_transform)

    return src_pcd_transformed, final_transform, icp_result.fitness, icp_result.inlier_rmse


def visualize_registration(source, target, transformed_source=None):

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0, 0])  # Đỏ
    target_temp.paint_uniform_color([0, 1, 0])  # Xanh lá

    if transformed_source is None:
        o3d.visualization.draw_geometries([source_temp, target_temp])
    else:
        transformed_source_temp = copy.deepcopy(transformed_source)
        transformed_source_temp.paint_uniform_color([0, 0, 1])  # Xanh dương
        o3d.visualization.draw_geometries([target_temp, transformed_source_temp])


def evaluate_registration(source, target, transformed_source, threshold=0.05):

    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    transformed_points = np.asarray(transformed_source.points)

    cd_bf = chamfer_distance(source_points, target_points)

    cd_af = chamfer_distance(transformed_points, target_points)

    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     transformed_source, target, threshold)

    metrics = {
        'chamfer_distance_before': cd_bf,
        # 'chamfer_s2t_before': cd_s2t_before,
        # 'chamfer_t2s_before': cd_t2s_before,
        'chamfer_distance_after': cd_af,
        # 'chamfer_s2t_after': cd_s2t_after,
        # 'chamfer_t2s_after': cd_t2s_after,
        # 'fitness': evaluation.fitness,  # Tỷ lệ điểm khớp trong ngưỡng
        # 'inlier_rmse': evaluation.inlier_rmse,  # RMSE của các điểm khớp
        # 'correspondence_set_size': len(evaluation.correspondence_set)  # Số lượng cặp điểm tương ứng
    }

    return metrics


def custom_partial_icp(source, target, max_iter=100, threshold=0.05):
    """
    ICP tùy chỉnh cho partial matching
    - source: đám mây điểm đầy đủ (hoặc visible part của nó)
    - target: đám mây điểm một phần (từ depth map)
    """
    # Khởi tạo
    result = copy.deepcopy(source)
    transformation = np.identity(4)
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # Tạo KD tree cho source (dùng để tìm điểm gần nhất)
    source_kdtree = o3d.geometry.KDTreeFlann(source)

    for iteration in range(max_iter):
        # 1. Tìm các cặp điểm tương ứng (chỉ từ target đến source)
        correspondences = []
        for i, point in enumerate(target_points):
            [_, idx, distance] = source_kdtree.search_knn_vector_3d(point, 1)
            if distance[0] < threshold ** 2:  # Chỉ lấy các điểm trong ngưỡng
                correspondences.append((i, idx[0]))

        if len(correspondences) < 3:  # Cần ít nhất 3 điểm để tính transformation
            break

        # 2. Trích xuất các điểm tương ứng
        p = np.zeros((len(correspondences), 3))  # Target points
        q = np.zeros((len(correspondences), 3))  # Source points

        for i, (target_idx, source_idx) in enumerate(correspondences):
            p[i] = target_points[target_idx]
            q[i] = source_points[source_idx]

        # 3. Tính transformation để minimize khoảng cách
        # (Sử dụng thuật toán SVD)
        p_centroid = np.mean(p, axis=0)
        q_centroid = np.mean(q, axis=0)

        p_centered = p - p_centroid
        q_centered = q - q_centroid

        # Tính ma trận hiệp phương sai
        H = np.dot(p_centered.T, q_centered)
        U, S, Vt = np.linalg.svd(H)

        # Tính ma trận xoay
        R = np.dot(Vt.T, U.T)

        # Đảm bảo R là ma trận xoay đúng (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Tính vector dịch chuyển
        t = q_centroid - np.dot(R, p_centroid)

        # 4. Tạo ma trận transformation
        current_transformation = np.identity(4)
        current_transformation[:3, :3] = R
        current_transformation[:3, 3] = t

        # 5. Cập nhật transformation tích lũy
        transformation = np.dot(current_transformation, transformation)

        # 6. Kiểm tra hội tụ
        error = np.mean(np.linalg.norm(np.dot(p, R.T) + t - q, axis=1))
        if error < threshold / 10:  # Hội tụ khi sai số đủ nhỏ
            break

        # 7. Cập nhật source points cho vòng lặp tiếp theo
        temp = np.ones((source_points.shape[0], 4))
        temp[:, :3] = source_points
        temp = np.dot(temp, current_transformation.T)
        source_points = temp[:, :3]

        # Cập nhật KD tree
        updated_source = o3d.geometry.PointCloud()
        updated_source.points = o3d.utility.Vector3dVector(source_points)
        source_kdtree = o3d.geometry.KDTreeFlann(updated_source)

    # Áp dụng transformation cuối cùng
    result.transform(transformation)

    # Tính fitness và inlier_rmse tương tự như hàm registration_icp
    # để có thể so sánh kết quả
    correspondences = []
    for i, point in enumerate(target_points):
        [_, idx, distance] = source_kdtree.search_knn_vector_3d(point, 1)
        if distance[0] < threshold ** 2:
            correspondences.append((i, idx[0]))

    fitness = len(correspondences) / len(target_points)
    squared_errors = []
    for i, j in correspondences:
        squared_errors.append(np.sum((target_points[i] - source_points[j]) ** 2))
    inlier_rmse = np.sqrt(np.mean(squared_errors))

    return result, transformation, fitness, inlier_rmse


def custom_robust_registration(source, target, threshold=0.05, max_iter=100):
    """
    Robust registration sử dụng TukeyLoss để giảm ảnh hưởng của outliers
    """
    # Khởi tạo
    current_transformation = np.identity(4)

    # Tạo TukeyLoss với scale là threshold
    loss = o3d.pipelines.registration.TukeyLoss(k=threshold)

    for i in range(max_iter):
        # Tạo một bản sao của source để áp dụng transformation hiện tại
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(current_transformation)

        # Thực hiện một bước ICP với robust kernel
        result_icp = o3d.pipelines.registration.registration_icp(
            source_transformed, target, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
            loss
        )

        # Cập nhật transformation
        current_transformation = np.dot(result_icp.transformation, current_transformation)

        # Kiểm tra hội tụ
        if np.allclose(result_icp.transformation, np.identity(4), atol=1e-6):
            break

    # Áp dụng transformation cuối cùng
    result = copy.deepcopy(source)
    result.transform(current_transformation)

    return result, current_transformation

def main(
        src_file,
        target_file,
        is_src_npy=True,
        is_target_npy=False,
        visualize=True,
        max_iter=100,
        threshold=0.02
):
    src_pcd = load_point_cloud(src_file, is_npy=is_src_npy)
    tar_pcd = load_point_cloud(target_file, is_npy=is_target_npy)

    source_normalized, src_centroid, src_maxdist = normalize_point_cloud(src_pcd)
    target_normalized, target_centroid, target_maxdist = normalize_point_cloud(tar_pcd)

    if not source_normalized.has_normals():
        print("Đang tính normal cho source...")
        source_normalized.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if not target_normalized.has_normals():
        print("Đang tính normal cho target...")
        target_normalized.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    print("Đang thực hiện ICP...")
    transformed_source, transform_matrix, fitness, inlier_rmse = icp_with_normal_init(
        source_normalized, target_normalized, max_iter=max_iter, threshold=threshold)

    transformed_source, transform_matrix = custom_robust_registration(
        transformed_source, target_normalized, max_iter=50, threshold=threshold)

    print(f"ICP hoàn thành với fitness: {fitness:.4f}, inlier RMSE: {inlier_rmse:.4f}")

    # Hiển thị kết quả đăng ký
    if visualize:
        print("Kết quả đăng ký:")
        visualize_registration(source_normalized, target_normalized, transformed_source)

    print("Đang đánh giá kết quả đăng ký...")
    metrics = evaluate_registration(source_normalized, target_normalized, transformed_source, threshold)

    print(metrics)

if __name__ == "__main__":
    source_file = "D:\\private\\objects_dataset_npy_10000\\objects\\a63e6333-b3b8-4487-b3ae-7c8c5e3092e8\\normalized_model.npy"
    target_file = "D:\\private\\scenes\\08df38e7-b9ec-40d1-8652-b1857959a6c7\\masked_cloud.ply"

    main(
        source_file,
        target_file,
        max_iter=10000,
        threshold=0.02
    )


