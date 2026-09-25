# Binary-only variant: installs pre-built executables instead of compiling.

cmake_minimum_required(VERSION 3.8)
project(swift_pico)

find_package(ament_cmake REQUIRED)

# --- Pre-built binaries -------------------------------------------------------
install(
  PROGRAMS config/mujoco_bridge
  RENAME mujoco_bridge
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/task_1b_controller
  RENAME task_1b_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  PROGRAMS src/task_1c_controller
  RENAME task_1c_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  FILES config/fastdds_profile.xml
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
