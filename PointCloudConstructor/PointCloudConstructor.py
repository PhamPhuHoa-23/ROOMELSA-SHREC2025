import os.path

import numpy as np
import open3d as o3d
import cv2
import os as sos

def depth_to_point_cloud(
        depth_map,
        rgb_image=None,
        output_file=None,
        mask=None,
        max_depth=10.0,
        filter_outliers=True,
        voxel_size=None
):
    """

    :param depth_map: The depth map (hxw)
    :param rgb_image: RGB image (hxwx3)
    :param output_file: Path to save the point cloud
    :param mask: Binary mask for valid depth values (Support for instance mask in query)
    :param max_depth: Maximum depth value
    :param filter_outliers: Whether to filter out outliers
    :param voxel_size: Voxel size for downsampling depth map
    :return: open3d.geometry.PointCloud
    """

    h, w = depth_map.shape[:2]
    print(h, w)
    if mask is None:
        mask = (depth_map > 0) & (depth_map < max_depth) & (~np.isnan(depth_map))

    points = []
    colors = []
    print(depth_map)
    print(depth_map.min(), depth_map.max())

    depth_map = depth_map / depth_map.max() * 1
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                r = depth_map[i, j]
                phi = (j / w) * 2 * np.pi
                theta = (i / h) * np.pi
                # r = 1

                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)

                points.append([x, y, z])

                if rgb_image is not None:
                    colors.append(rgb_image[i, j] / 255)

    points = np.array(points)
    colors = np.array(colors)

    if (len(points) == 0):
        raise ValueError("No points found")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        print("alo", len(colors), len(points))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if filter_outliers and len(points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=2, std_ratio=0.1)

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if output_file is not None:
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Saved point cloud to {output_file}")

    return pcd


def visualize_point_cloud(
        pcd,
        window_name="Yokoso to my entertainment duel"
):
    """

    :param pcd: Point Cloud
    :param window_name: Window name
    :return:
    """
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    o3d.visualization.draw_geometries([pcd], window_name=window_name)

# if __name__ == "__main__":
#     fol_path = "G:\\My Drive\\public_data\\1e6013d0-dc3a-421e-8b56-c85d2a2a9706"
#     depth_map = np.load(sos.path.join(fol_path, "0_colors_depth.npy"))  # Load your depth map
#     rgb_image = cv2.imread(os.path.join(fol_path, "0_colors.jpg"))
#     if rgb_image is not None:
#         rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
#
#     # Convert to point cloud
#     pcd = depth_to_point_cloud(
#         depth_map=depth_map,
#         rgb_image=rgb_image,
#         output_file="point_cloud.ply",
#         filter_outliers=True,
#         # voxel_size=0.01  # For downsampling
#     )
#
#     # Visualize
#     visualize_point_cloud(pcd)

def depth_to_mesh(depth_map, rgb_image=None, output_file=None,
                  visualize=True, depth_scale=1.0,
                  reconstruction_method='alpha_shape'):
    """
    Convert a panoramic depth map to a mesh.

    Args:
        depth_map (numpy.ndarray): The depth map (height x width)
        rgb_image (numpy.ndarray, optional): RGB image for coloring the mesh
        output_file (str, optional): Path to save the mesh (.ply or .obj)
        visualize (bool, optional): Whether to visualize the mesh
        depth_scale (float, optional): Scale factor for depth values
        reconstruction_method (str, optional): Method to use for reconstruction:
            'alpha_shape', 'ball_pivoting', or 'poisson'

    Returns:
        open3d.geometry.TriangleMesh: The generated mesh
    """
    # Get dimensions
    h, w = depth_map.shape

    # Create a valid mask
    mask = (depth_map > 0) & (~np.isnan(depth_map))

    # Sample fewer points for better performance
    # This creates a more uniform but sparser point cloud
    stride = 2  # Skip pixels for better performance

    points = []
    colors = []

    # First, create a more dense and uniform point cloud
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            if mask[i, j]:
                # Get depth value and apply scale
                r = depth_map[i, j] * depth_scale

                # Convert to spherical coordinates - adjusted formula
                theta = (float(i) / float(h)) * np.pi  # Vertical angle [0, π]
                phi = (float(j) / float(w)) * 2 * np.pi  # Horizontal angle [0, 2π]

                # Convert to Cartesian coordinates
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)

                points.append([x, y, z])

                # Add color if RGB image is provided
                if rgb_image is not None:
                    if len(rgb_image.shape) == 3:  # RGB image
                        colors.append(rgb_image[i, j] / 255.0)  # Normalize to [0, 1]

    # Convert to numpy arrays
    points = np.array(points)
    colors = np.array(colors) if len(colors) > 0 else None

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals - essential for meshing algorithms
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Create mesh from point cloud
    mesh = None

    if reconstruction_method == 'alpha_shape':
        print("Performing Alpha Shape reconstruction...")
        # Alpha shape reconstruction (faster but less smooth)
        alpha = 0.5  # Adjust this parameter for your point cloud density
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()

    elif reconstruction_method == 'ball_pivoting':
        print("Performing Ball Pivoting reconstruction...")
        # Ball pivoting (good for evenly distributed points)
        radii = [0.05, 0.1, 0.2, 0.4]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        mesh.compute_vertex_normals()

    elif reconstruction_method == 'poisson':
        print("Performing Poisson reconstruction...")
        # Poisson reconstruction (best for closed surfaces with good normals)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False)

        # The mesh may contain artifacts outside the point cloud
        # Use vertex densities to clean it up
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.compute_vertex_normals()

    else:
        raise ValueError(f"Unknown reconstruction method: {reconstruction_method}")

    if mesh is None or len(mesh.triangles) == 0:
        print("Failed to create mesh. Falling back to point cloud visualization.")
        if output_file:
            o3d.io.write_point_cloud(output_file, pcd)
        return pcd

    # Transfer colors from point cloud to mesh if available
    if colors is not None:
        # A simple strategy - paint vertices with the nearest point's color
        # For more sophisticated texture mapping, you would need UV coordinates
        if not mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.zeros((len(mesh.vertices), 3)))

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for i, vertex in enumerate(mesh.vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
            if len(idx) > 0:
                mesh.vertex_colors[i] = pcd.colors[idx[0]]

    # Save mesh to file if specified
    if output_file:
        o3d.io.write_triangle_mesh(output_file, mesh)
        print(f"Mesh saved to: {output_file}")

    # Visualize if requested
    if visualize:
        visualize_mesh(mesh)

    return mesh


def visualize_mesh(mesh, point_cloud=None, window_name="Mesh Viewer"):
    """
    Visualize a mesh using Open3D.

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to visualize
        point_cloud (open3d.geometry.PointCloud): Optional point cloud to show alongside mesh
        window_name (str): Window name
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=1280, height=720)

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)

    # Add mesh
    vis.add_geometry(mesh)

    # Add point cloud if provided
    if point_cloud is not None:
        vis.add_geometry(point_cloud)

    # Improve rendering options
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = True
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    render_option.light_on = True

    # Setup camera
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)

    # Reset view
    vis.reset_view_point(True)

    print("\nTương tác với mô hình 3D:")
    print("- Xoay: Click chuột trái + kéo")
    print("- Pan: Shift + Click chuột trái + kéo")
    print("- Zoom: Cuộn chuột")
    print("- Bấm 'h' để xem thêm trợ giúp")

    # # Run visualizer
    vis.run()
    vis.destroy_window()


import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def create_dense_mesh(depth_map, rgb_image=None, output_file=None,
                      method='poisson_dense', texture_mapping=True,
                      visualize=True, depth_scale=1.0):
    """
    Create a realistic dense mesh reconstruction from a panoramic depth map.

    Args:
        depth_map (numpy.ndarray): Depth map (height x width)
        rgb_image (numpy.ndarray): RGB image for texturing
        output_file (str): Path to save the mesh (.ply or .obj)
        method (str): Reconstruction method ('poisson_dense', 'voxel_carving')
        texture_mapping (bool): Whether to apply texture mapping
        visualize (bool): Whether to visualize the result
        depth_scale (float): Scale factor for depth values

    Returns:
        open3d.geometry.TriangleMesh: The generated mesh
    """
    # Get dimensions
    h, w = depth_map.shape

    # Create a valid mask
    mask = (depth_map > 0) & (~np.isnan(depth_map))

    # Create a cleaner depth map by filling holes and smoothing
    cleaned_depth = depth_map.copy()
    cleaned_depth[~mask] = 0

    # Apply median filter to remove noise
    cleaned_depth = cv2.medianBlur(cleaned_depth.astype(np.float32), 5)

    # Fill small holes using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # Create dense point cloud
    print("Creating dense point cloud...")

    points = []
    colors = []
    normals = []

    # Control point density - lower stride means more points
    stride = 1  # Dense sampling

    # For tracking progress
    total_pixels = (h // stride) * (w // stride)
    processed = 0
    last_percent = 0

    # Create organized point cloud (preserving structure helps with reconstruction)
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            # Update progress
            processed += 1
            percent = (processed * 100) // total_pixels
            if percent > last_percent and percent % 10 == 0:
                print(f"Processing points: {percent}% complete")
                last_percent = percent

            if mask_closed[i, j]:
                # Get depth value and apply scale
                r = cleaned_depth[i, j] * depth_scale

                if r <= 0:
                    continue

                # Convert to spherical coordinates
                theta = (float(i) / float(h)) * np.pi - np.pi / 2 # Vertical angle [0, π]
                phi = (float(j) / float(w)) * 2 * np.pi  - np.pi # Horizontal angle [0, 2π]

                # Convert to Cartesian coordinates
                x = 1 * np.sin(theta) * np.cos(phi)
                y = 1 * np.sin(theta) * np.sin(phi)
                z = 1 * np.cos(theta)

                points.append([x, y, z])

                # Add color if RGB image is provided
                if rgb_image is not None:
                    if len(rgb_image.shape) == 3:  # RGB image
                        colors.append(rgb_image[i, j] / 255.0)  # Normalize to [0, 1]

                # Calculate normal vector (pointing from center to the point)
                normal = np.array([x, y, z])
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                    normals.append(normal)

    # Convert to numpy arrays
    points = np.array(points)
    colors = np.array(colors) if len(colors) > 0 else None
    normals = np.array(normals) if len(normals) > 0 else None

    if len(points) == 0:
        raise ValueError("No valid points generated. Check your depth map and mask.")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None and len(normals) > 0:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        # Estimate normals if not provided
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Create mesh based on selected method
    mesh = None

    if method == 'poisson_dense':
        print("Creating dense mesh using Poisson reconstruction...")

        # Better parameters for Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, width=0, scale=1.1, linear_fit=False)

        # Clean the mesh based on density
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Simplify mesh if it's too complex
        triangle_count = len(mesh.triangles)
        if triangle_count > 1000000:  # If more than 1M triangles
            print(f"Simplifying mesh from {triangle_count} triangles...")
            mesh = mesh.simplify_quadric_decimation(100000)
            print(f"Simplified to {len(mesh.triangles)} triangles")

    elif method == 'voxel_carving':
        print("Creating voxel-based reconstruction...")

        # Convert point cloud to voxel grid
        voxel_size = 0.05  # Adjust depending on scale
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

        # Create a mesh from the voxel grid using marching cubes
        voxels = np.asarray(voxel_grid.get_voxels())
        if len(voxels) > 0:
            # Create a 3D grid
            x_min, y_min, z_min = np.min(points, axis=0) - voxel_size * 2
            x_max, y_max, z_max = np.max(points, axis=0) + voxel_size * 2

            # This process would actually require a complete voxel carving algorithm
            # with surface extraction via marching cubes
            # Since O3D doesn't directly support this, we'll use the voxel grid for visualization
            # and use Poisson for the actual mesh

            print("Falling back to Poisson reconstruction for actual mesh...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False)

            # Clean the mesh based on density
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(vertices_to_remove)

            # For demonstration, we'll also show the voxel grid
            if visualize:
                vis = o3d.visualization.Visualizer()
                vis.create_window("Voxel Grid Visualization", width=1280, height=720)
                vis.add_geometry(voxel_grid)
                vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
                vis.run()
                vis.destroy_window()
        else:
            print("No voxels created. Falling back to Poisson reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    if mesh is None or len(mesh.triangles) == 0:
        print("Failed to create mesh. Returning point cloud instead.")
        if output_file:
            o3d.io.write_point_cloud(output_file, pcd)
        return pcd

    # Compute vertex normals
    mesh.compute_vertex_normals()

    # Apply texture mapping if RGB is provided and requested
    if rgb_image is not None and texture_mapping:
        # A more sophisticated approach to texture mapping
        if not mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.zeros((len(mesh.vertices), 3)))

        # Use nearest neighbors for better color mapping
        mesh_vertices = np.asarray(mesh.vertices)
        if len(points) > 0 and len(mesh_vertices) > 0:
            print("Applying texture to mesh...")

            # Using k-nearest neighbors for better texture mapping
            nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points)
            distances, indices = nn.kneighbors(mesh_vertices)

            for i in range(len(mesh_vertices)):
                if i < len(indices) and indices[i][0] < len(colors):
                    mesh.vertex_colors[i] = colors[indices[i][0]]

    # Smooth the mesh to make it more realistic
    print("Smoothing mesh...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)

    # Save mesh to file if specified
    if output_file:
        o3d.io.write_triangle_mesh(output_file, mesh)
        print(f"Mesh saved to: {output_file}")

    # Visualize if requested
    if visualize:
        visualize_realistic_mesh(mesh, pcd)

    return mesh


def visualize_realistic_mesh(mesh, point_cloud=None):
    """
    Visualize a mesh in a more realistic rendering style.

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to visualize
        point_cloud (open3d.geometry.PointCloud, optional): Original point cloud
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Realistic Room Reconstruction", width=1280, height=720)

    # Add small coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_frame)

    # Add mesh
    vis.add_geometry(mesh)

    # Setup renderer for more realistic visualization
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = False
    render_option.mesh_show_back_face = False
    render_option.light_on = True
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    render_option.point_size = 1.0

    # Add lighting for better 3D appearance
    render_option.light_on = True

    # Enable mesh shading for more realistic look
    # render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.SmoothShade

    # Setup camera
    view_control = vis.get_view_control()

    # Show UI instructions
    print("\nTương tác với mô hình 3D thực tế:")
    print("- Xoay: Click chuột trái + kéo")
    print("- Pan: Shift + Click chuột trái + kéo hoặc click chuột phải + kéo")
    print("- Zoom: Cuộn chuột hoặc Ctrl + Click chuột trái + kéo lên/xuống")
    print("- Bấm 'h' để xem thêm trợ giúp")
    print("- Bấm 'l' để hiển thị/ẩn lưới (wireframe)")
    print("- Bấm 'p' để chuyển giữa các chế độ hiển thị")

    # Run visualizer
    vis.run()
    vis.destroy_window()


def create_room_view(depth_map, rgb_image, view_type='3d', output_file=None):
    """
    Create a more intuitive view of the room from depth map.

    Args:
        depth_map (numpy.ndarray): Depth map
        rgb_image (numpy.ndarray): RGB image
        view_type (str): Type of visualization ('3d', 'topdown', 'floorplan')
        output_file (str, optional): Path to save visualization

    Returns:
        numpy.ndarray: Visualization image
    """
    h, w = depth_map.shape

    if view_type == 'topdown':
        # Create a top-down view (like a floor plan)
        print("Creating top-down view...")

        # Create point cloud
        points = []
        colors = []

        for i in range(0, h, 1):
            for j in range(0, w, 1):
                if depth_map[i, j] > 0:
                    # Get depth value
                    r = depth_map[i, j]

                    # Convert to 3D point
                    # theta = (float(i) / float(h)) * np.pi
                    # phi = (float(j) / float(w)) * 2 * np.pi
                    #
                    # x = r * np.sin(theta) * np.cos(phi)
                    # y = r * np.sin(theta) * np.sin(phi)
                    # z = r * np.cos(theta)
                    theta = (j / w) * 2 * np.pi
                    phi = (i / h) * np.pi

                    x = 1 * np.sin(phi) * np.sin(theta)
                    y = 1 * np.cos(phi)
                    z = 1 * np.sin(phi) * np.cos(theta)

                    points.append([x, y, z])
                    if rgb_image is not None:
                        colors.append(rgb_image[i, j])

        if len(points) > 0:
            points = np.array(points)
            colors = np.array(colors) if len(colors) > 0 else None

            # Create top-down view
            plt.figure(figsize=(10, 10))
            plt.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=0.5)
            plt.axis('equal')
            plt.title('Top-Down View (Floor Plan)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)

            if output_file:
                plt.savefig(output_file)
                print(f"Top-down view saved to: {output_file}")

            plt.show()

            # Create a more detailed floor plan
            grid_size = 0.1
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

            grid_width = int((x_max - x_min) / grid_size) + 1
            grid_height = int((y_max - y_min) / grid_size) + 1

            grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

            for point in points:
                x, y = point[0], point[1]
                grid_x = int((x - x_min) / grid_size)
                grid_y = int((y - y_min) / grid_size)

                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    grid[grid_y, grid_x] = 255

            # Apply dilation to enhance walls
            kernel = np.ones((3, 3), np.uint8)
            grid = cv2.dilate(grid, kernel, iterations=1)

            # Save and display the floor plan
            if output_file:
                floorplan_file = os.path.splitext(output_file)[0] + "_floorplan.png"
                cv2.imwrite(floorplan_file, grid)
                print(f"Floor plan saved to: {floorplan_file}")

            plt.figure(figsize=(10, 10))
            plt.imshow(grid, cmap='gray')
            plt.title('Floor Plan')
            plt.axis('off')
            plt.show()

            return grid

    elif view_type == 'floorplan':
        # Create a floor plan style visualization
        # This is a more sophisticated approach that would require
        # actual floor plan generation algorithms
        print("Full floor plan generation requires more complex algorithms.")
        print("Falling back to top-down view...")
        return create_room_view(depth_map, rgb_image, 'topdown', output_file)

    else:  # '3d' or default
        # Create a 3D mesh and visualize it
        mesh = create_dense_mesh(depth_map, rgb_image, output_file,
                                 method='poisson_dense', visualize=True)
        return mesh


if __name__ == '__main__':
    fol = "21bce930-fd0c-4ef3-be2d-6cdd5bb3a8c7"
    fol_path = f"G:\\My Drive\\public_data\\{fol}"
    depth_map = np.load(sos.path.join(fol_path, "0_colors_depth2.npy"))  # Load your depth map
    rgb_image = cv2.imread(os.path.join(fol_path, "0_colors.png"))
    print(rgb_image)
    if rgb_image is not None:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Convert to point cloud
    # mesh = depth_to_mesh(
    #     depth_map=depth_map,
    #     rgb_image=rgb_image,
    #     visualize=True
    # )
    #
    # # Visualize
    # visualize_mesh(mesh)

    pcd = depth_to_point_cloud(
        depth_map=depth_map,
        rgb_image=rgb_image,
        output_file="point_cloud.ply",
        filter_outliers=False,
        # voxel_size=0.01  # For downsampling
    )

    # Visualize
    visualize_point_cloud(pcd)

    # print("\n1. Creating realistic 3D mesh...")
    # mesh = create_dense_mesh(
    #     depth_map=depth_map,
    #     rgb_image=rgb_image,
    #     output_file="realistic_room.ply",
    #     method='poisson_dense',
    #     texture_mapping=True
    # )
    #
    # # Create room visualization (top-down view)
    # print("\n2. Creating top-down view (floor plan)...")
    # create_room_view(
    #     depth_map=depth_map,
    #     rgb_image=rgb_image,
    #     view_type='topdown',
    #     output_file="room_topdown.png"
    # )