# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

DISABLE_IF_OPENCV_NOT_FOUND()
DISABLE_IF_CUDA_NOT_FOUND()

# This is the name of the executable
set(EXECUTABLE_NAME 09_OPENCV_VIDEO_TEMPLATE)

# In UNIX, set PROJECT_SOURCES using the following command
# find . | grep .cpp
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_SOURCES 
"app.cpp"
"main.cpp"
"mat_viewer.cpp"
"ui.cpp"
"video_viewer.cpp" 
)

# In UNIX, set PROJECT_HEADERS using the following command
# find . | grep .h
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_HEADERS
"app.hpp"
"main.hpp"
"mat_viewer.hpp"
"ui.hpp"
"util/vec2.hpp"
"util/vec3.hpp"
"video_viewer.hpp")

# In UNIX, set PROJECT_HEADERS using the following command
# find . | grep .cu
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
 set(PROJECT_CUDA
"algorithms/cuda_algorithm.cuh"
"algorithms/cuda_algorithm.cu"
"algorithms/dummy/dummy_cuda_algorithm.cuh"
"algorithms/dummy/dummy_cuda_algorithm.cu"
 )


# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_CUDA})


# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glad glfw imgui ${OpenCV_LIBS}) 


# Tests
add_executable(T09_TEST_CUDA_CV "testMatToGLTex.cu")
target_link_libraries (T09_TEST_CUDA_CV glad glfw ${OpenCV_LIBS}) 


