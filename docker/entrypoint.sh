#!/bin/bash

# Source existing bashrc
source /opt/ros/humble/setup.bash
source ~/.bashrc

export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH

# Check if the workspace is built
if [ ! -d "/ros_ws/build" ]; then
    echo "Building the ROS workspace"
    cd /ros_ws && colcon build
fi

# Source the workspace
source /ros_ws/install/setup.bash

# Run the entrypoint
exec "$@"