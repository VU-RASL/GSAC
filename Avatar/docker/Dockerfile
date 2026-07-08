FROM nvidia/cuda:12.8.0-devel-ubuntu20.04

ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && apt-get install -y \
    build-essential \
    rsync \
    curl \
    wget \
    htop \
    git \
    openssh-server \
    nano \
    cmake \
    unzip \
    zip \
    vim \
    ffmpeg \
    tmux \
    freeglut3-dev \
    fish \
    ninja-build \
    libglm-dev

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Addign the missing cudagl layers
RUN apt-get update && apt-get install -y \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev

ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics"

ENV PYOPENGL_PLATFORM=egl

RUN ls /usr/share/glvnd/egl_vendor.d/ || true

COPY docker/EGL.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

## glew installation from source
RUN curl -L https://downloads.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0.tgz > /tmp/glew-2.1.0.tgz
RUN mkdir -p /tmp && \
    cd /tmp && tar zxf /tmp/glew-2.1.0.tgz && cd glew-2.1.0 && \
    SYSTEM=linux-egl make && \
    SYSTEM=linux-egl make install && \
    rm -rf /tmp/glew-2.1.0.zip /tmp/glew-2.1.0

# fixuid # may not work with latest (if not root user already exits)
ARG USERNAME=user
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME

# miniconda
WORKDIR /home/$USERNAME
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/$USERNAME/miniconda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda


# python libs
RUN pip install --upgrade pip

RUN echo 113
COPY docker/environment.yml /home/$USERNAME/environment.yml

RUN conda env create -f /home/$USERNAME/environment.yml

RUN echo "source activate gsac" > ~/.bashrc
ENV PATH="/home/$USERNAME/miniconda/envs/gsac/bin:$PATH"

RUN conda info --envs


# Pin CUDA 12.8 PyTorch stack before installing CUDA extensions.
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

RUN echo 1
RUN pip install git+https://github.com/karfly/nvdiffrast_compute-capability_6.0
RUN pip install git+https://github.com/DmitryUlyanov/yamlenv.git


# install nvdiffrast
COPY 3rd_party/nvdiffrast/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install --upgrade pip
RUN pip install imageio imageio-ffmpeg

COPY 3rd_party/nvdiffrast/nvdiffrast /tmp/nvdiffrast/
COPY 3rd_party/nvdiffrast/README.md 3rd_party/nvdiffrast/setup.py /tmp/
RUN cd /tmp && pip install .


# Append Missing Packages from environment.yml file

# 1. Install lightweight utilities and wrappers (OpenAI CLIP moved to Git source)
RUN pip install --no-cache-dir \
    "gsplat==1.4.0" \
    ipykernel \
    jupyter-client \
    "opencv-python-headless==4.10.0.84" \
    "graphviz==0.8.4" \
    "torch-fidelity==0.3.0" \
    "torchtyping==0.1.5" \
    "trimesh==4.8.1" \
    "triton==3.4.0" \
    "typeguard==2.13.3"

# Tyro installation spearately
RUN pip install --no-cache-dir "tyro==0.9.31" --ignore-installed

# 2. Sequenced OpenMMLab installation.
# Build mmcv from source so CUDA ops are  available.
RUN pip install --no-cache-dir openmim setuptools && \
    mim install "mmengine==0.10.7" && \
    pip uninstall -y mmcv mmcv-lite mmcv-full || true && \
    pip cache purge && \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    LD_LIBRARY_PATH=/home/user/miniconda/envs/gsac/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
    pip install --no-cache-dir -v \
        "git+https://github.com/open-mmlab/mmcv.git@v2.1.0" \
        --no-build-isolation && \
    mim install "mmdet==3.2.0" && \
    mim install "mmpose==1.3.2"


# 3. Native Source Builds for Research Frameworks (Including OpenAI CLIP)
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/facebookresearch/segment-anything-2.git
RUN pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
RUN pip install "nerfacc==0.5.2" --no-build-isolation
RUN TCNN_CUDA_ARCHITECTURES=89 pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch --no-build-isolation
RUN pip install git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.18.0 --no-build-isolation

# Build PyTorch3D from source with CUDA support
RUN pip uninstall -y pytorch3d || true && \
    FORCE_CUDA=1 \
    LD_LIBRARY_PATH=/home/user/miniconda/envs/gsac/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
    pip install --no-cache-dir -v \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    --no-build-isolation

# Final NumPy/OpenCV installation
RUN pip uninstall -y numpy opencv-contrib-python opencv-python opencv-python-headless opencv-contrib-python-headless || true && \
    pip install --no-cache-dir "numpy==1.23.5" && \
    pip install --no-cache-dir "opencv-python==4.11.0.86" && \
    pip install --no-cache-dir "opencv-python-headless==4.10.0.84" && \
    pip install --no-cache-dir --no-deps "opencv-contrib-python==4.13.0.92"

# This is the only version that we found. No 2.2.2.0 found. (maybe it was a mistaake)
RUN pip install --no-cache-dir --no-deps "human-body-prior==0.8.5.0"

# Installing the diff_gaussian_rasterization (no need to install anything in postbuild)
COPY --chown=user:user 3rd_party/diff_gaussian_rasterization-alphadep /tmp/diff_gaussian_rasterization-alphadep
RUN pip install --no-cache-dir --no-build-isolation \
    /tmp/diff_gaussian_rasterization-alphadep && \
    rm -rf /tmp/diff_gaussian_rasterization-alphadep

# docker setup
WORKDIR /
ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]


# ============================================================
# Git dependency references used when this Docker image worked
# (captured June 2026)
#
# PyTorch3D stable:
#   75ebeeaea0908c5527e7b1e305fbc7681382db47
#
# OpenAI CLIP HEAD:
#   d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
#
# Segment Anything 2 HEAD:
#   2b90b9f5ceec907a1c18123530e92e794ad901a4
#
# lang-segment-anything HEAD:
#   918043ed4666eea04da88aa179eb8d27ef4b1a1d
#
# tiny-cuda-nn HEAD:
#   749dd70c5afc5a9dadb85e5652ed65d55e0ba187
#
# NOTE:
# These hashes are recorded for reproducibility.
# Current Dockerfile intentionally installs the tagged/branch
# versions that were verified to work. If exact reproducibility
# is required in the future, replace the git refs with these
# commit hashes.
# ============================================================
