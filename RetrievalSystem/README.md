# RetrievalSystem

A web-based visualization and retrieval system for 3D shapes and related text queries using Qdrant vector database.

## Overview

The RetrievalSystem provides a web interface for exploring and querying a vector database of 3D shapes and related text. It offers:

- Visual exploration of text query embeddings
- Search across multiple collection types (text, image, shape)
- Combined search with weighted scoring from different modalities
- Distance-based scoring for shape similarity using Chamfer distance
- Export of search results to CSV format

This system is particularly useful for exploration, evaluation, and demonstration of 3D object retrieval systems.

## Features

- Interactive web UI for exploring vector embeddings
- Multi-modal search across text, image, and shape collections
- Configurable weighting between different modalities
- Visualization of search results with metadata
- CSV export for evaluation and sharing
- REST API for programmatic access

## Setup

### Requirements

```bash
# Create conda environment
conda create -n roomelsa python=3.9
conda activate roomelsa

# Install dependencies
pip install flask flask-cors qdrant-client numpy pillow

# Install Qdrant
# Option 1: Docker (recommended)
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v /path/to/qdrant_storage:/qdrant/storage qdrant/qdrant

# Option 2: Install locally
# Follow instructions at https://qdrant.tech/documentation/guides/installation/
```

### Preparing Qdrant Collections

Before using the application, you need to set up your Qdrant collections. You can use the provided `Qdrant.ipynb` notebook to:

1. Create collections for text, image, and shape embeddings
2. Upload embeddings to the respective collections
3. Test basic queries

## Running the Application

```bash
# Navigate to the RetrievalSystem directory
cd RetrievalSystem

# Start the Flask server
python app.py
```

The application will be available at http://localhost:6333.

## Using the Web Interface

1. **Browse Queries:** The main page displays all available text queries. You can use the search box to filter queries.
2. **Select a Query:** Click on a query to see its search results.
3. **Configure Search:** Select the query type (combined, image, or shape) and adjust weights if needed.
4. **View Results:** The results are displayed with scores and thumbnail images.
5. **Export Results:** Use the export buttons to download results in CSV format.

## Score Calculation

The system uses a sophisticated scoring mechanism that combines:

- Cosine similarity between vectors for text, image, and shape
- Chamfer distance for direct shape-to-shape comparison
- Weighted combination

The weighted combination are implemented in `score/calculate_score.py`.