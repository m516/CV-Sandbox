# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# This is the name of the executable
set(EXECUTABLE_NAME 06_RAYTRACING_IN_ONE_WEEKEND)

# In UNIX, set PROJECT_SOURCES using the following command
# find . | grep .cpp
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_SOURCES 
"app.cpp"
"main.cpp"
"raytracing/cpu_raytracer_1.cpp"
"raytracing/cpu_raytracer_2.cpp"
"raytracing/simple_cpu_renderer.cpp"
"raytracing/util/primitives/sphere.cpp"
"renderer.cpp"
"ui.cpp"
 "raytracing/util/hittable_list.cpp")

# In UNIX, set PROJECT_HEADERS using the following command
# find . | grep .h
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_HEADERS
"app.h"
"main.h"
"raytracing/cpu_raytracer_1.h"
"raytracing/cpu_raytracer_2.h"
"raytracing/generic_cpu_raytracer.h"
"raytracing/raytracers.h"
"raytracing/raytracing_utils.h"
"raytracing/simple_cpu_renderer.h"
"raytracing/util/camera.h"
"raytracing/util/hittable.h"
"raytracing/util/primitives/hittable_list.h"
"raytracing/util/primitives/sphere.h"
"raytracing/util/ray.h"
"raytracing/util/rot_math.h"
"raytracing/util/vec2.h"
"raytracing/util/vec3.h"
"renderer.h"
"ui.h"
 "raytracing/util/hittable_list.h")


# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glfw imgui) 