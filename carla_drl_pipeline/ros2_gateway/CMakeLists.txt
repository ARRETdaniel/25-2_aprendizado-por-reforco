cmake_minimum_required(VERSION 3.8)
project(carla_ros2_bridge)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# Find ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable
add_executable(carla_bridge_node
  src/carla_bridge_node.cpp
  src/message_converters.cpp
)

# Link dependencies
ament_target_dependencies(carla_bridge_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  cv_bridge
  image_transport
)

target_link_libraries(carla_bridge_node
  ${OpenCV_LIBRARIES}
  ${ZMQ_LIBRARIES}
)

target_include_directories(carla_bridge_node PUBLIC
  ${ZMQ_INCLUDE_DIRS}
)

target_compile_options(carla_bridge_node PUBLIC
  ${ZMQ_CFLAGS_OTHER}
)

# Install targets
install(TARGETS
  carla_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  
  # Run linters
  ament_lint_auto_find_test_dependencies()
  
  # Add unit tests
  ament_add_gtest(test_message_converters
    test/test_message_converters.cpp
    src/message_converters.cpp
  )
  
  ament_target_dependencies(test_message_converters
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
  )
endif()

ament_package()
