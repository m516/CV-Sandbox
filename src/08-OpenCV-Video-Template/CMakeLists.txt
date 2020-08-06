# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

DISABLE_IF_OPENCV_NOT_FOUND()
DISABLE_IF_CUDA_NOT_FOUND()

# This is the name of the executable
set(EXECUTABLE_NAME 08_OPENCV_VIDEO_TEMPLATE)

# In UNIX, set PROJECT_SOURCES using the following command
# find . | grep .cpp
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_SOURCES 
"app.cpp"
"main.cpp"
"mat_viewer.cpp"
"ui.cpp"
 "video_viewer.cpp" 
 "algorithms/optical_flow/optical_flow_ui.cpp"
)

# In UNIX, set PROJECT_HEADERS using the following command
# find . | grep .h
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_HEADERS
"app.h"
"main.h"
"mat_viewer.h"
"ui.h"
"util/vec2.h"
"util/vec3.h"
"video_viewer.h"
 "algorithms/optical_flow/optical_flow_ui.h" 
 "algorithms/optical_flow/optical_flow.h")

 set(PROJECT_CUDA
"algorithms/optical_flow/optical_flow.cpp"
 )


# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_CUDA})

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glfw imgui ${OpenCV_LIBS}) 