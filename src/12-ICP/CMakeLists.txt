# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# This is the name of the executable
set(EXECUTABLE_NAME 12_ICP)

# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} "main.cpp" "main.hpp"  "pointcloud/pointcloud.hpp" "pointcloud/pointcloud.cpp" "pointcloud/renderer.hpp" "pointcloud/renderer.cpp"   "app/window.hpp" "app/window.cpp")

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glm glad glfw imgui)