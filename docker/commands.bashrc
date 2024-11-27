# Configuration file for ROS2 and custom aliases
# Gets exported to .bashrc within the container

# Source ROS2 Humble setup file
source /opt/ros/humble/setup.bash

# Set ROS2 console output format to include severity and node name
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}] [{name}]: {message}'

# Enable colored console output for ROS2 messages
export RCUTILS_COLORIZED_OUTPUT=1

# Alias to kill all ROS2 and Gazebo processes
alias tek_kill="( \
    pkill -f ros; \
    pkill -f gzserver; \
    pkill -f gzclient \
)"

# Alias to build the workspace and source the new setup file
alias tek_install='colcon build && source install/setup.bash'

# Zsh like experience
bind 'set completion-ignore-case on'
bind 'set show-all-if-ambiguous on'
bind 'TAB:menu-complete'

# Add Conditional Source of ROS2 workspace setup file
if [ -f "/ros_ws/install/setup.bash" ]; then
    source /ros_ws/install/setup.bash
fi

# For Pinocchio for Franka
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH # Adapt your desired python version here
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH