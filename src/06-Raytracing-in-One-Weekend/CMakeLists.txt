# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# This is the name of the executable
set(EXECUTABLE_NAME 06_RAYTRACING_IN_ONE_WEEKEND)

set(PROJECT_SOURCES 
"main.cpp"
"app.cpp"
"ui.cpp"
"renderer.cpp" 
"raytracing/simpleCPURenderer.cpp" 
"raytracing/raytracers/CPUraytracer1.cpp"
)

set(PROJECT_HEADERS
"main.h"
"app.h"
"ui.h"
"renderer.h" 
"raytracing/raytracingutils.h"
"raytracing/util/vec2.h"
"raytracing/util/vec3.h"
"raytracing/simpleCPURenderer.h"
 "raytracing/raytracers/CPUraytracer1.h" 
 "raytracing/raytracers/genericCPUtracer.h" 
 "raytracing/util/hittable.h" 
 "raytracing/util/ray.h" "raytracing/raytracers/raytracers.h")

# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glfw imgui)