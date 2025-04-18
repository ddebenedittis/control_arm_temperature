# =========================== Project Configuration ============================

cmake_minimum_required(VERSION 3.8)
project(robot_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================= Find Dependencies ==============================

find_package(ament_cmake REQUIRED)

# ============================ Install Directories =============================

install(DIRECTORY config launch params worlds DESTINATION share/${PROJECT_NAME})

# =============================================================================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
