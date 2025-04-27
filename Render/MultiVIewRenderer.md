# MultiViewRenderer

A class for rendering 3D models from multiple viewpoints using PyTorch3D.

## Overview

The `MultiViewRenderer` class provides functionality to:

- Load 3D mesh models from OBJ files
- Automatically normalize and center meshes
- Render meshes from multiple predefined viewpoints
- Save rendered images with consistent naming conventions
- Adjust camera and lighting parameters for optimal visualization

The renderer is designed to create consistent multi-view renderings of 3D objects, which can be used for training deep learning models or for visualization purposes.

## Features

- Configurable rendering parameters (image size, camera positions, lighting)
- Custom 12-view configuration with 3 elevations × 4 azimuths
- Automatic camera distance calculation based on model size
- Support for texture mapping
- Fallback texture generation for untextured models
- Efficient batch processing of multiple models

## Setup

### Requirements

```bash
# Create conda environment
conda create -n roomelsa python=3.9
conda activate roomelsa

# Install PyTorch with CUDA support
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# Install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install other dependencies
pip install numpy pillow tqdm
```

## Usage

### Basic Usage

```python
from Render.MultiViewRenderer import MultiViewRenderer
import torch

# Initialize renderer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = MultiViewRenderer(
    device=device,
    image_size=224,
    custom_views=True,  # Use default 12-view configuration
    auto_distance=True,
    distance_margin=1.2,
    lighting_intensity=1.0
)

# Load and render a mesh
obj_path = "path/to/model.obj"
texture_path = "path/to/texture.png"  # Optional
output_dir = "path/to/output"

# Save multiple views
image_paths = renderer.save_rendered_images(
    obj_path=obj_path,
    output_dir=output_dir,
    uuid1="category_id",
    uuid2="model_id",
    texture_path=texture_path
)

# Or render single view
mesh = renderer.load_mesh(obj_path, texture_path)
image = renderer.render_mesh(mesh, azimuth=45.0, elevation=30.0)
```

### Using with main.py

The repository includes a `main.py` script for batch processing multiple models:

```bash
python main.py \
    --data_root /path/to/dataset \
    --output_dir /path/to/output \
    --json_path object.json \
    --image_size 600 \
    --auto_distance True \
    --lighting_intensity 1.0 \
    --background_color 1,1,1
```

## Parameters

- **image_size**: Size of the rendered images (square)
- **custom_views**: Whether to use custom 12-view configuration (3 elevations × 4 azimuths)
- **num_views**: Number of viewpoints to render (only used if custom_views=False)
- **elevation**: Camera elevation in degrees (only used if custom_views=False)
- **distance**: Camera distance from the object (used as default or minimum if auto_distance=True)
- **auto_distance**: Whether to automatically calculate optimal camera distance based on model size
- **distance_margin**: Margin multiplier for auto-distance calculation (higher = more space around model)
- **background_color**: Background color (R, G, B) with values in [0, 1]
- **lighting_intensity**: Intensity of the lighting (affects ambient, diffuse and specular)