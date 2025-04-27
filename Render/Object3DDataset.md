# Object3DDataset

A PyTorch Dataset class for loading and managing 3D object models from a hierarchical directory structure.

## Overview

The `Object3DDataset` class provides a standardized interface for accessing 3D models and their associated data (textures, images, descriptions) organized in a hierarchical directory structure. It's designed to work with PyTorch's DataLoader for efficient batch processing of 3D data.

## Features

- Loads 3D models from a structured directory hierarchy
- Supports both normalized and raw model variants
- Handles associated metadata (textures, images, descriptions)
- Compatible with PyTorch's DataLoader for efficient batch processing
- Provides direct access to models by UUID
- Customizable with transform functions

## Directory Structure

The dataset expects a directory structure like:

```
data_root/
├── uuid1/
│   ├── uuid2/
│   │   ├── normalized_model.obj
│   │   ├── raw_model.obj (optional)
│   │   ├── model.mtl
│   │   ├── texture.png
│   │   ├── image.jpg
│   │   └── description.txt (optional)
│   └── ...
└── ...
```

And a metadata JSON file (`object.json`) containing information about each model:

```json
{
  "uuid1": {
    "uuid2": {
      "obj_path": "uuid1/uuid2/normalized_model.obj",
      "mtl_path": "uuid1/uuid2/model.mtl",
      "texture_path": "uuid1/uuid2/texture.png",
      "image_path": "uuid1/uuid2/image.jpg",
      "description": "..."
    }
  }
}
```

## Setup

### Requirements

```bash
# Create conda environment
conda create -n roomelsa python=3.9
conda activate roomelsa

# Core dependencies
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# Install other dependencies
pip install numpy pillow
```

## Usage

### Basic Usage

```python
from Render.Object3DDataset import Object3DDataset
from torch.utils.data import DataLoader

# Initialize dataset
dataset = Object3DDataset(
    data_root="/path/to/dataset",
    json_path="object.json",
    model_type="normalized"  # or "raw"
)

# Access a single item
item = dataset[0]
print(f"UUID1: {item['uuid1']}")
print(f"UUID2: {item['uuid2']}")
print(f"OBJ path: {item['obj_path']}")

# Access a specific item by UUID
item = dataset.get_item_by_uuid("uuid1", "uuid2")

# Use with DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
for batch in dataloader:
    # Process batch
    pass
```

### With Transforms

```python
import torchvision.transforms as transforms
from PIL import Image

# Define a transform function
def custom_transform(item):
    # Load and transform image
    if "image_path" in item and os.path.exists(item["image_path"]):
        image = Image.open(item["image_path"]).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        item["image_tensor"] = transform(image)
    return item

# Initialize dataset with transform
dataset = Object3DDataset(
    data_root="/path/to/dataset",
    json_path="object.json",
    transform=custom_transform
)
```

## Class Methods

- **`__init__`**: Initialize the dataset with paths and options
- **`__len__`**: Return the number of items in the dataset
- **`__getitem__`**: Get an item at a specific index
- **`get_item_by_uuid`**: Get an item by its UUID1 and UUID2

## Parameters

- **data_root**: Root directory of the dataset
- **json_path**: Path to the metadata JSON file relative to data_root
- **model_type**: Type of model to load ("normalized" or "raw")
- **transform**: Optional transform function to apply to the data

## Return Format

Each item returned by the dataset is a dictionary containing:

```python
{
    "uuid1": "...",                # First-level UUID
    "uuid2": "...",                # Second-level UUID
    "obj_path": "/path/to/model.obj",  # Full path to OBJ file
    "mtl_path": "/path/to/model.mtl",  # Full path to MTL file
    "texture_path": "/path/to/texture.png",  # Full path to texture file
    "image_path": "/path/to/image.jpg",  # Full path to image file
    "description": "..."           # Description text (if available)
}
```