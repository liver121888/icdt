# Use CUDA image as base
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /ros_ws

# Install essential dependencies and common tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    lsb-release \
    gnupg2 \
    curl \
    build-essential \
    tzdata \
    ranger \
    git \
    neovim \
    iputils-ping \
    net-tools \
    dnsutils \
    htop \
    nvtop \
    wget \
    unzip \
    gnome-terminal \
    tmux \
    locales \
    apt-transport-https && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Setup ROS 2 Humble repository and install ROS 2 Humble (Desktop Full Installation)
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb [trusted=yes] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' && \
    apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-rosdep2 \
    python3-pip \
    python3-nose \
    python3-colcon-common-extensions \
    python3-vcstool \
    python-is-python3

# Install ROS 2 control, controllers, and visualization tools
RUN apt-get install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-control-toolbox \
    ros-humble-joint-state-publisher \
    ros-humble-rviz2 \
    ros-humble-tf-transformations

# Install MoveIt and Gazebo ROS 2 control
RUN apt-get install -y \
    ros-humble-moveit \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-setup-assistant \
    ros-humble-gazebo-ros2-control \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-pick-ik \
    vim-gtk3 \
    ament-cmake-nose

# Setup and install Intel RealSense libraries
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y \
    librealsense2-dkms \
    librealsense2-utils \
    ros-humble-librealsense2* \
    ros-humble-realsense2-camera

# Install torch
RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install Transformers for OwlViT
RUN pip install -q git+https://github.com/huggingface/transformers.git

# Install Remaining Dependencies
RUN pip install open3d==0.18.0
RUN pip install opencv-python==4.9.0.80
RUN pip install transforms3d==0.4.2
RUN apt install -y ros-humble-tf-transformations
RUN pip install numpy==1.24.2

# Initialize rosdep
# RUN rosdep init
RUN rosdep update

# For Arduino
RUN pip install pyserial

# Source ROS 2 setup file
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Install x11 dependencies for GUI applications
RUN apt-get install -y libgl1-mesa-glx libxrender1 libxext6 libxtst6

# Install OpenAI
RUN pip install openai==1.53.0

# For Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh
RUN pip install ollama
RUN pip install streamlit

# For whisper.cpp
RUN apt-get install -y libportaudio2 
RUN pip install sounddevice
RUN pip install pynput

# Install Franka dependencies
RUN apt-get update && apt-get install -y \
    bc \
    ca-certificates \
    ros-humble-ament-cmake-clang-format \
    ros-humble-aruco-ros \
    libssl-dev \
    libelf-dev \
    libpoco-dev \
    libeigen3-dev \
    dpkg \
    bison \
    flex \
    dwarves \
    zstd \
    libncurses-dev \
    liburdfdom-headers-dev \
    libconsole-bridge-dev \
    libtinyxml2-dev \
    libignition-gazebo6-dev

# Add the robotpkg repository
RUN curl -sSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | tee /etc/apt/keyrings/robotpkg.asc > /dev/null && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list

# Install Pinocchio library from robotpkg
RUN apt-get update && apt-get install -y robotpkg-pinocchio

# Remove any existing libfranka installations to avoid conflicts
RUN apt-get remove -y "*libfranka*"

# Clone the libfranka repository and prepare the build directory
RUN git clone --recursive https://github.com/frankaemika/libfranka.git /libfranka && \
    cd /libfranka && \
    git checkout 0.14.1 && \
    git submodule update --init --recursive && \
    mkdir build

# Set working directory to the build directory
WORKDIR /libfranka/build

# Set CMAKE_PREFIX_PATH explicitly to ensure proper configuration
ENV CMAKE_PREFIX_PATH=/opt/openrobots/lib/cmake

# Configure, build, and package libfranka
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. && \
    cmake --build . && \
    cpack -G DEB

# Install the generated Debian package
RUN dpkg -i libfranka*.deb

RUN git clone https://github.com/ggerganov/whisper.cpp /whispercpp && \
    cd /whispercpp && \
    sh ./models/download-ggml-model.sh base.en && \
    make -j4

# Return to ROS workspace for default container entry
WORKDIR /ros_ws

# Copy custom commands into .bashrc
COPY docker/commands.bashrc /ros_ws/
RUN cat /ros_ws/commands.bashrc >> ~/.bashrc

# Set the default command to run when starting the container
COPY docker/entrypoint.sh /ros_ws/
ENTRYPOINT ["./entrypoint.sh"]
CMD ["/bin/bash"]
