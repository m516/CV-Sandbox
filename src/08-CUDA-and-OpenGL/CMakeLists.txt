# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

DISABLE_IF_CUDA_NOT_FOUND()

# This is the name of the executable
set(EXECUTABLE_NAME 08_CUDA_AND_OPENGL)

# In UNIX, set PROJECT_SOURCES using the following command
# find . | grep .cpp
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_SOURCES
#Place source files here
)

# In UNIX, set PROJECT_HEADERS using the following command
# find . | grep .h
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_HEADERS
"cuda_check_error.h"
)

 set(PROJECT_CUDA
"main.cu"
 )


# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_CUDA})

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glad glfw) 