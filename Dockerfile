FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -ay

# Create conda environment
RUN conda create -n roomelsa python=3.9 -y && \
    echo "source activate roomelsa" > ~/.bashrc

# Activate conda environment for subsequent RUN commands
SHELL ["conda", "run", "-n", "roomelsa", "/bin/bash", "-c"]

# Install PyTorch and related packages
RUN conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y && \
    conda install -c iopath iopath -y && \
    conda install numpy scipy matplotlib pandas scikit-learn -c conda-forge -y

# Install pip packages
RUN pip install --no-cache-dir \
    open3d \
    omegaconf \
    flask \
    flask-cors \
    qdrant-client \
    pillow \
    tqdm \
    open-clip-torch \
    huggingface_hub \
    sympy \
    simpleicp \
    einops \
    timm \
    wandb \
    dgl==1.1.2 \
    torch_redstone \
    open_clip_torch

# Install PyTorch3D from source
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install OpenShape as editable package
RUN cd OpenShape_code && pip install -e .

# Expose ports for Flask and Qdrant
EXPOSE 5000 6333 6334

# Set default command
CMD ["/bin/bash"]

