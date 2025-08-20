cmake_minimum_required(VERSION 3.5)
project(ros2_gateway)

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
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Find ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Find jsoncpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZMQ_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})

# Create executable
add_executable(gateway_node src/gateway_node.cpp)

# Target dependencies
ament_target_dependencies(gateway_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  OpenCV
)

# Link libraries
target_link_libraries(gateway_node
  ${OpenCV_LIBRARIES}
  ${ZMQ_LIBRARIES}
  ${JSONCPP_LIBRARIES}
)

# Install targets
install(TARGETS
  gateway_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
