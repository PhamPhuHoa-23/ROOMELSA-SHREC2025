import os
import sys
import torch
import numpy as np

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

def normalize_pc(pc):
    """Chuẩn hóa point cloud: trừ centroid và chia cho max norm."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_norm = np.max(np.linalg.norm(pc, axis=1))
    if max_norm > 1e-6:
        pc = pc / max_norm
    return pc

def parse_mtl_for_kd(mtl_path):
    """Đọc file .mtl và lấy màu Kd mặc định."""
    kd_color = [0.5, 0.5, 0.5]  # Mặc định nếu không tìm thấy Kd
    try:
        with open(mtl_path, 'r') as f:
            for line in f:
                if line.startswith('Kd'):
                    kd_values = line.split()[1:]
                    kd_color = [float(v) for v in kd_values]
                    break
    except Exception as e:
        print(f"Lỗi khi đọc file .mtl {mtl_path}: {e}")
    return np.array(kd_color, dtype=np.float32)


def load_obj_to_pointcloud(obj_path, num_points=2048, normalize=True):
    """
    Load file .obj, convert to point cloud and subsample to a specific number of points with PyTorch3D.
    Returns a tensor with 6 dimensions, including xyz coordinates and RGB colors.

    Args:
        obj_path (str): Path to the .obj file
        num_points (int): Number of points to sample from the mesh
        normalize (bool): Whether to normalize the point cloud to unit sphere

    Returns:
        torch.Tensor: Point cloud with shape (num_points, 6) where each point contains:
                     - xyz: 3D coordinates (first 3 dimensions)
                     - rgb: RGB color values (last 3 dimensions)
    """
    # Load the obj file
    mesh = load_objs_as_meshes([obj_path], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Sample points from the mesh
    point_cloud = sample_points_from_meshes(mesh, num_points)  # (1, num_points, 3)
    point_cloud = point_cloud.squeeze(0)  # (num_points, 3)

    # Get colors for each point (either from texture or use default)
    colors = torch.ones_like(point_cloud)  # Default white color

    # if has_texture:
    #     # This is a simplified approximation - for accurate texture mapping,
    #     # you would need to implement barycentric interpolation
    #     # This is complex and beyond the scope of this example
    #     # For this example, we'll use a simple approximation with dummy colors
    #     # based on normalized position
    colors = (point_cloud - point_cloud.min(0)[0]) / (point_cloud.max(0)[0] - point_cloud.min(0)[0])

    # Normalize point cloud to unit sphere if requested
    if normalize:
        center = point_cloud.mean(0)
        point_cloud = point_cloud - center
        scale = torch.max(point_cloud.norm(dim=1))
        point_cloud = point_cloud / scale

    # Concatenate xyz coordinates with rgb colors
    point_cloud_with_color = torch.cat([point_cloud, colors], dim=1)  # (num_points, 6)

    return point_cloud_with_color

if __name__ == "__main__":
    input_dir = 'G:/My Drive/public_data'  # Thư mục chứa dataset
    output_dir = 'G:/My Drive/public_data_numpy_10000'  # Thư mục lưu file .npy
    num_points = 10000

    with open("G:/My Drive/public_data/object.json", "r") as f:
        d = f.read()
        import json

        d = json.loads(d)

    obj_files = []
    for key, value in d.items():
        for key2, value2 in value.items():
            obj_files.append(os.path.join(input_dir, value2['obj_path'].replace("\\", "/")))

    skipped_count = 0
    processed_count = 0
    
    for obj_path in tqdm(obj_files, desc="Processing .obj files"):
        try:
            # Tạo đường dẫn lưu file .npy
            relative_path = os.path.relpath(obj_path, input_dir)
            npy_path = os.path.join(output_dir, relative_path).replace('.obj', '.npy')
            
            # Kiểm tra xem file .npy đã tồn tại chưa
            if os.path.exists(npy_path):
                skipped_count += 1
                continue  # Bỏ qua file đã được xử lý
                
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)

            # Load và chuyển đổi point cloud
            pointcloud = load_obj_to_pointcloud(obj_path, num_points=num_points, normalize=True).cpu().numpy()
            if pointcloud is None:
                continue

            # Lưu thành file .npy
            np.save(npy_path, pointcloud)
            processed_count += 1
            print(f"Đã lưu: {npy_path}")

        except Exception as e:
            print(f"Lỗi khi xử lý file {obj_path}: {e}")
    
    print(f"Kết quả: {skipped_count} files đã được bỏ qua, {processed_count} files đã được xử lý")