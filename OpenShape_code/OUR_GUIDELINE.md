# OpenShape Fine-tuning Guide

This guide will walk you through the process of setting up and fine-tuning OpenShape models with your custom datasets. OpenShape is a 3D foundation model that bridges 3D shapes with other modalities such as text and images.

## Environment Setup

First, let's create a conda environment and install the required dependencies:

```bash
conda create -n roomelsa python=3.9
conda activate roomelsa

# Core dependencies
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# Install DGL with CPU version to avoid compatibility issues
pip install dgl==1.1.2

# Install other dependencies
pip install numpy open3d wandb omegaconf einops timm tqdm
pip install torch_redstone

# Install Open CLIP
pip install open_clip_torch
```

> **Note**: The code imports many libraries, but not all of them are always needed. The listed packages above are the core requirements for running the training process.

## Data Preparation

The CustomDataset in `data.py` supports two ways to provide data:

1. Using a pair list file
2. Automatically finding pairs from two directories

### Option 1: Using a pair list file

Create a text file with each line containing the path to a point cloud file and its corresponding text embedding file, separated by a space:

```
/path/to/pc1.npy /path/to/text_embed1.npy
/path/to/pc2.npy /path/to/text_embed2.npy
...
```

### Option 2: Using separate directories

Place your point cloud files in a directory structure:
```
/pc_dir/uuid1/uuid2/normalized_model.npy
```

And text embeddings in another directory:
```
/text_embed_dir/uuid1_uuid2.npy
```

The code will automatically match them based on UUIDs.

### Point Cloud Format

Your point cloud files should be numpy arrays containing both position and color information:
- First 3 columns: XYZ coordinates
- Last 3 columns: RGB values (normalized between 0-1)

## Configuration

The configuration files are in YAML format and located in the `src/configs/` directory. Two important files to note are:

1. `custom_train.yaml`: Basic configuration for custom training
2. `train_p40.yaml`: Alternative configuration that can also be used as reference

### Key Configuration Parameters

Here are the key sections to modify in your configuration file:

```yaml
model:
  name: PointBERT        # Model architecture (e.g., PointBERT, DGCNN, PointNeXt)
  scaling: 4             # Model size scaling factor
  use_dense: True        # Whether to use dense representation
  in_channel: 6          # Input channels (3 for XYZ, 6 for XYZ+RGB)
  out_channel: 1280      # Output embedding dimension

training:
  max_epoch: 50          # Training epochs
  lr: 0.0005             # Learning rate
  use_text_contras: True # Use text contrastive loss
  use_img_contras: False # Use image contrastive loss

custom_dataset:
  # Option 1: List of paired files
  pair_list: "/path/to/pairs.txt"  
  
  # Option 2: Auto-find pairs from directories
  pc_dir: "/path/to/point_clouds"
  text_embed_dir: "/path/to/text_embeddings"
  
  # Dataset parameters
  num_points: 10000      # Number of points to sample
  y_up: True             # If true, swap Y and Z axes (for different coordinate systems)
  normalize: True        # Normalize point clouds
  random_z_rotate: True  # Apply random rotation around Z axis for augmentation
  use_color: True        # Use RGB color information
```

## Training

To start training, run:

```bash
python src/main.py --config src/configs/custom_train.yaml --trial_name your_experiment_name
```

Additional command line arguments:
- `--resume /path/to/checkpoint.pt`: Resume training from a checkpoint
- `--autoresume`: Automatically resume from the latest checkpoint
- `--ngpu N`: Set the number of GPUs to use (default: 1)
- `--exp_dir path/to/save`: Directory to save experiment results

This fine-tuning guide is built upon the [OpenShape](https://github.com/Colin97/OpenShape) project. Please cite their work if you use this code for research.