# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# This is the name of the executable
set(EXECUTABLE_NAME 03_HELLO_IMGUI_GLFW)

# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} "main.cpp" "main.h")

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glfw imgui)