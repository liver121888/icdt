cmake_minimum_required(VERSION 3.5)

project(whisper_ros)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# Specify the path to whisper.cpp directory
set(WHISPER_CPP_PATH "/path/to/whisper.cpp")  # Example: "/home/username/whisper.cpp"

# Include the whisper.cpp header files
include_directories(${WHISPER_CPP_PATH})

# Add your ROS 2 node executable
add_executable(whisper_node src/whisper_node.cpp)

# Link the whisper.cpp static library or object file to your ROS 2 node
# Assuming whisper.cpp builds into a static library (libwhisper.a) or a shared library (libwhisper.so)
target_link_libraries(whisper_node ${WHISPER_CPP_PATH}/libwhisper.a)

# Link to ROS 2 dependencies
ament_target_dependencies(whisper_node rclcpp)

# Install the node
install(TARGETS whisper_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
