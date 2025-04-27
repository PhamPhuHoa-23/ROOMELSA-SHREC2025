# ROOMELSA-SHREC2025

A 3D object retrieval system for room elements and shapes, designed for the SHREC 2025 challenge.

## Overview

This repository contains a comprehensive system for 3D shape analysis, retrieval, and visualization. The system supports:

- Multi-view rendering of 3D models
- Point cloud construction from depth maps
- Point cloud-to-mesh conversion
- Shape retrieval based on text queries
- Shape similarity calculation using Chamfer distance
- Web-based visualization of retrieval results

## Components

- **MultiViewRenderer**: Renders multiple viewpoints from 3D models using PyTorch3D
- **Object3DDataset**: Dataset class for loading and managing 3D objects
- **DataCreator**: Utilities for creating and processing datasets
- **RetrievalSystem**: Web-based visualization and search system
- **Utils**: Various utility scripts for processing and analyzing 3D data

## Setup

Please see the individual component READMEs for specific setup instructions.

### General Requirements

```bash
# Create conda environment
conda create -n roomelsa python=3.9
conda activate roomelsa

# Core dependencies
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c iopath iopath
conda install numpy scipy matplotlib pandas scikit-learn -c conda-forge
pip install open3d omegaconf
pip install flask flask-cors qdrant-client
pip install pillow tqdm open-clip-torch huggingface_hub sympy simpleicp
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
cd OpenShape_code
pip install -e .
```

## Usage

See the individual component READMEs for specific usage instructions.
