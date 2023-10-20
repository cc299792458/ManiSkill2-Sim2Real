FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES all

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python=3.8 && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# RUN pip install mani-skill2==0.4.2 && pip cache purge
RUN pip install setuptools==63.2.0
# RUN pip install "stable-baselines3[extra]<2.0.0"
RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
RUN pip install pytorch3d 
RUN pip install ninja
COPY requirements.py /opt/conda/lib/python3.8/site-packages/wheel/vendored/packaging/requirements.py
RUN git clone https://github.com/cc299792458/ManiSkill2-Sim2Real.git # clip return of stackcube
RUN cd ManiSkill2-Sim2Real && pip install -e .
RUN rm -d ManiSkill2-Sim2Real/ManiSkill2-Learn
RUN git clone https://github.com/cc299792458/ManiSkill2-Learn.git # change evaluation part
RUN cd ManiSkill2-Learn && git checkout ms2_gym && pip install -e .
# RUN ln -s ManiSkill2-Sim2real/data ManiSkill2-Learn/data
RUN pip cache purge

