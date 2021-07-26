FROM osrf/ros:noetic-desktop-full
# choose this image if you are working on a MacBook M1
# FROM arm64v8/ros:noetic   

WORKDIR /root/ws/
COPY . .

# INSTALL ROS AND OTHER DEPENDENCIES
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN --mount=type=cache,target=/var/cache/apt,id=apt \
  apt-get update && apt-get install -yq --no-install-recommends \
  ros-${ROS_DISTRO}-ros-tutorials \
  ros-${ROS_DISTRO}-common-tutorials \
  python3-pip \
  git-all \
  flake8 && \
  rm -rf /var/lib/apt/lists/*

# CACHING FOR PIP 
# Note: This makes re-building of images a LOT quicker since we do not need to
# download all pip packages again and again.
ENV PIP_CACHE_DIR=/root/.cache/pip
RUN mkdir -p $PIP_CACHE_DIR

# INSTALL TORCH
# Note: doing this via requiremnts.txt always downloads the GPU version, which we
# do not need at the moment and requires signifanctly more disk space.
RUN --mount=type=cache,target=$PIP_CACHE_DIR,id=pip \
  python3 -m pip install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# INSTALL PYTHON PACKAGES
RUN python3 -m pip install --upgrade pip
RUN --mount=type=cache,target=$PIP_CACHE_DIR,id=pip \
  python3 -m pip install -r src/bayesopt4ros/requirements.txt 

# EXTEND BASHRC
RUN echo "source /root/ws/devel_isolated/setup.bash" >> ~/.bashrc
