# Render

## Overview

The Render module, primarily through `Object.ipynb`, creates a standardized JSON dataset format that serves as the backbone for the entire ROOMELSA-SHREC2025 system. This JSON structure is critical for organizing 3D models in a hierarchical structure that other components can efficiently access and process.

## Object.ipynb

The `Object.ipynb` notebook is responsible for:

1. Scanning a hierarchical directory of 3D models
2. Creating a standardized JSON metadata file that describes the dataset
3. Establishing the relationships between models, textures, and images

The generated JSON file is the central configuration that connects all system components together.

## JSON Structure

The created JSON file follows this structure:

```json
{
  "uuid1": {
    "uuid2": {
      "obj_path": "uuid1/uuid2/normalized_model.obj",
      "texture_path": "uuid1/uuid2/texture.png",
      "image_path": "uuid1/uuid2/image.jpg",
      "description": "..."
    }
  }
}
```

This structure maps directly to a directory hierarchy:

```
data_root/
├── uuid1/
│   ├── uuid2/
│   │   ├── normalized_model.obj
│   │   ├── texture.png
│   │   ├── image.jpg
│   │   └── description.txt (optional)
```

## Integration with Other Components

### MultiViewRenderer

The MultiViewRenderer relies on this JSON structure to:
- Locate the 3D models (.obj files)
- Find associated textures
- Save rendered views while preserving the hierarchical organization
- Use the UUID structure for consistent naming of output files

```python
# Example of how MultiViewRenderer uses the JSON metadata
renderer.save_rendered_images(
    obj_path=item_data["obj_path"],
    output_dir=output_dir,
    uuid1=item_data["uuid1"],
    uuid2=item_data["uuid2"],
    texture_path=item_data["texture_path"]
)
```

### Object3DDataset

The Object3DDataset class directly consumes this JSON structure to:
- Create a PyTorch Dataset for efficient data loading
- Access models, textures, and images consistently
- Support both normalized and raw model variants
- Enable direct access to models by UUID

```python
# Object3DDataset initializes from this JSON
dataset = Object3DDataset(
    data_root="/path/to/dataset",
    json_path="object.json",
    model_type="normalized"
)

# Models can be accessed by UUID
item = dataset.get_item_by_uuid("uuid1", "uuid2")
```

### Main Rendering Script (main.py)

The main.py script relies on the JSON structure to:
- Find all models to be processed
- Track which models have already been rendered
- Filter models by UUID when needed
- Maintain a consistent output structure

```python
# main.py uses Object3DDataset, which loads the JSON
dataset = Object3DDataset(
    data_root=args.data_root,
    json_path=args.json_path,
    model_type=args.model_type
)

# Process each model in the dataset
for item_data in dataset:
    uuid1 = item_data["uuid1"]
    uuid2 = item_data["uuid2"]
    obj_path = item_data["obj_path"]
    texture_path = item_data["texture_path"]
    
    # Create output directory maintaining hierarchy
    output_dir = os.path.join(args.output_dir, uuid1, uuid2)
```

## Setup

### Requirements

```bash
# Create conda environment
conda create -n roomelsa python=3.9
conda activate roomelsa

# Install required packages
pip install numpy pandas tqdm
```

## Usage

To create the dataset JSON:

1. Open `Object.ipynb` in Jupyter Notebook/Lab
2. Set the `DATASET_PATH` variable to your dataset root directory
3. Execute the notebook cells to create the JSON file

```python
# In Object.ipynb
DATASET_PATH = 'path/to/your/dataset'

# Execute the create_dataset_json function
create_dataset_json(DATASET_PATH, "object.json", find_description=False)
```

This will:
1. Scan all directories under DATASET_PATH
2. Find OBJ files, textures, images, and descriptions
3. Create a structured JSON file with relative paths
4. Save the output as "object.json"
