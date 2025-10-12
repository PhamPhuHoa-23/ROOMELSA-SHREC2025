# ROOMELSA-SHREC2025

A 3D object retrieval system for room elements and shapes, designed for the SHREC 2025 challenge.

## Overview

This project, ROOMELSA-SHREC2025, is a comprehensive 3D object retrieval system designed for the SHREC 2025 challenge. The primary goal is to retrieve relevant 3D models of room elements and furniture from a large database based on natural language text queries. The system provides an end-to-end solution, from data processing and feature extraction to a web-based interface for interactive retrieval and visualization.

## Methodology

The methodology is divided into three main stages: Data Preparation, Feature Extraction, and Retrieval.

### 1. Data Preparation

3D models are first converted into a format suitable for feature extraction. This involves two steps:

- **Multi-View Rendering**: Each 3D object is rendered from multiple viewpoints to capture its geometric and textural information from different angles. This process generates a set of 2D images for each 3D model.
- **Point Cloud Generation**: The rendered images and their corresponding depth maps are used to generate colored point clouds, which serve as the primary 3D representation for our feature extraction model. Each point in the cloud has both spatial coordinates (X, Y, Z) and color information (R, G, B).

### 2. Feature Extraction

We employ a deep learning model from the OpenShape project to learn a joint embedding space for 3D shapes and text.

- **Model Architecture**: The system uses a point cloud-based encoder, such as PointBERT or PointNeXt, to transform a 3D point cloud \\(P\\) into a high-dimensional feature vector \\(v_P \in \mathbb{R}^d\\). Simultaneously, a pre-trained text encoder (from OpenCLIP) is used to convert a text query \\(T\\) into a feature vector \\(v_T \in \mathbb{R}^d\\) of the same dimension.

- **Training**: The model is trained using a contrastive learning approach. Given a set of \\(N\\) pairs of point clouds and their corresponding text descriptions \\(\\{(P*i, T_i)\\}*{i=1}^N\\), the objective is to minimize the contrastive loss. The similarity between a point cloud \\(P*i\\) and a text query \\(T_j\\) is measured by the cosine similarity of their embeddings, \\(s(i, j) = \frac{v*{P*i} \cdot v*{T*j}}{\\|v*{P*i}\\| \\|v*{T_j}\|}\\).

The training objective is to maximize the similarity of corresponding pairs \\((P_i, T_i)\\) while minimizing the similarity of non-corresponding pairs. This is achieved using the InfoNCE loss, which is a form of cross-entropy loss. For a given point cloud \\(P_i\\), the loss is:

\\[ \mathcal{L}_{P_i} = -\log \frac{\exp(s(i, i) / \tau)}{\sum_{j=1}^N \exp(s(i, j) / \tau)} \\]

where \\(\tau\\) is a temperature parameter that controls the sharpness of the distribution. A symmetric loss is calculated for the text query \\(T_i\\), and the total loss is the average of the two:

\\[ \mathcal{L}_{\text{total}} = \frac{1}{2N} \sum_{i=1}^N (\mathcal{L}_{P_i} + \mathcal{L}_{T_i}) \\]

### 3. Retrieval

The retrieval process is facilitated by a vector database and a web-based user interface.

- **Indexing**: The extracted feature vectors for all 3D models in the database are indexed and stored in a Qdrant vector database for efficient similarity search.

- **Querying**: When a user enters a text query, it is encoded into a feature vector using the same text encoder used during training. This query vector is then used to search the Qdrant database to find the 3D models with the most similar feature vectors, based on cosine similarity.

- **Ranking**: The retrieved models are ranked based on their similarity scores and presented to the user through a web interface.

## Evaluation Method

The performance of the retrieval system is evaluated using a combination of similarity metrics. The final score for a retrieved object is a weighted combination of multiple similarity measures:

- **Cosine Similarity**: The primary metric for retrieval is the cosine similarity between the query text embedding and the 3D shape embedding in the learned joint feature space.

- **Chamfer Distance**: For more fine-grained shape similarity assessment, we use the Chamfer Distance. It measures the distance between two point clouds, \\(P_1\\) and \\(P_2\\), and is defined as:

\\[ d_{CD}(P_1, P_2) = \frac{1}{|P_1|} \sum_{x \in P_1} \min_{y \in P_2} \\|x - y\\|_2^2 + \frac{1}{|P_2|} \sum_{y \in P_2} \min_{x \in P_1} \\|x - y\\|_2^2 \\]

To improve the accuracy of the Chamfer Distance, point clouds are first aligned using Principal Component Analysis (PCA) and the Iterative Closest Point (ICP) algorithm.

- **Combined Score**: The final ranking score is a weighted sum of the cosine similarity and a score derived from the Chamfer distance. The `advanced_scoring` function in `RetrievalSystem/score/calculate_score.py` implements this logic:

\\[ \text{Score} = w_1 \cdot \text{sim}_{\text{cosine}}(v_T, v_P) + w_2 \cdot f(d_{CD}(P_T, P_P)) \\]

where \\(P_T\\) is the point cloud of the query shape (if applicable) and \\(P_P\\) is the point cloud of the retrieved shape, and \\(f(\cdot)\\) is a function that converts the Chamfer distance to a similarity score.

## How to Run the Project

### 1. Setup Environment

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

### 2. Data Preparation

1.  Organize your 3D models in a directory.
2.  Use the `Render/Object.ipynb` notebook to create a JSON file that catalogs your dataset.
3.  Run the `Render/main.py` script to perform multi-view rendering of the 3D models. This will generate images and depth maps.
4.  Convert the rendered outputs to point clouds. You can use the `ObjectToNpy.py` script for this.

### 3. Training (Fine-tuning OpenShape)

1.  Prepare your dataset of point clouds and corresponding text embeddings as described in `OpenShape_code/OUR_GUIDELINE.md`.
2.  Configure your training in `OpenShape_code/src/configs/custom_train.yaml`.
3.  Start training by running:
    ```bash
    python OpenShape_code/src/main.py --config OpenShape_code/src/configs/custom_train.yaml
    ```

### 4. Indexing Data in Qdrant

1.  Set up and run a Qdrant instance (Docker is recommended).
2.  Use the `OpenShapeInference.py` script to extract features for all your 3D models.
3.  Use the `RetrievalSystem/Qdrant.ipynb` notebook to create collections in Qdrant and upload the extracted feature vectors.

### 5. Running the Retrieval System

1.  Start the Flask server:
    ```bash
    cd RetrievalSystem
    python app.py
    ```
2.  Open your web browser and navigate to `http://localhost:5000` (or the configured port) to use the retrieval system.
