# Utils

A collection of utility scripts for processing, analyzing, and manipulating 3D data.

## Overview

This directory contains a variety of utility scripts that support various operations related to 3D data processing, including:

- Chamfer distance calculation between point clouds
- Mask extraction from images
- Color picking tools
- Masking tool with GUI interface

These utilities are designed to work with the other components of the ROOMELSA-SHREC2025 system.

## Components

### ChamferDistance.py

A module for calculating Chamfer distance between point clouds, which is a measure of similarity between two sets of points.

#### Features:
- Point cloud normalization
- Point cloud loading from various formats
- PCA alignment for initial registration
- ICP (Iterative Closest Point) alignment
- Chamfer distance calculation
- Visualization of point cloud registration

#### Usage:
```python
import numpy as np
import open3d as o3d
from Utils.ChamferDistance import load_point_cloud, normalize_point_cloud, pca_alignment, chamfer_distance, visualize_registration

# Load point clouds
source_pcd = load_point_cloud("source.ply", is_npy=False)
target_pcd = load_point_cloud("target.ply", is_npy=False)

# Normalize point clouds
source_pcd, src_centroid, src_scale = normalize_point_cloud(source_pcd)
target_pcd, tgt_centroid, tgt_scale = normalize_point_cloud(target_pcd)

# Visualize before alignment
visualize_registration(source_pcd, target_pcd)

# Align using PCA
source_pcd, transform = pca_alignment(source_pcd, target_pcd)

# Create point cloud objects
source_npy = source_pcd.points
target_npy = target_pcd.points
source_pcd1 = PointCloud(source_npy, columns=["x", "y", "z"])
target_pcd1 = PointCloud(target_npy, columns=["x", "y", "z"])

# Calculate Chamfer Distance
icp = SimpleICP()
icp.add_point_clouds(target_pcd1, source_pcd1)
H, source_pcd_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)
transformed_pcd = o3d.geometry.PointCloud()
transformed_pcd.points = o3d.utility.Vector3dVector(source_pcd_transformed)

# Visualize after alignment
chamfer_dist = chamfer_distance(transformed_pcd, target_pcd)
print(f"Chamfer Distance: {chamfer_dist}")

visualize_registration(source_pcd, target_pcd, transformed_pcd)
```

### ColorPicker.py

A simple utility for picking color values from images using OpenCV.

#### Features:
- Interactive color selection
- RGB value display
- Multiple color sampling

#### Usage:
```bash
python ColorPicker.py
```

### mask_tool.py

A GUI tool for creating and editing masks on images.

#### Features:
- Multiple drawing tools (bounding box, freehand, polygon)
- Color-based masking
- Mask saving and loading
- Batch processing capabilities

#### Usage:
```bash
python mask_tool.py /path/to/images_directory
```