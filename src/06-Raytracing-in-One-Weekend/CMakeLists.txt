# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# This is the name of the executable
set(EXECUTABLE_NAME 06_RAYTRACING_IN_ONE_WEEKEND)

# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} "main.cpp" "main.h" "app.cpp" "app.h" "ui.cpp" "ui.h" "main.h")

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glfw imgui)