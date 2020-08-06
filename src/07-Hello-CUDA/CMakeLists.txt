# CMakeLists.txt
disable_if_cuda_not_found()

# This is the name of the executable
set(EXECUTABLE_NAME 07_HELLO_CUDA)

set(PROJECT_CUDA
HelloCUDA.cu
)

set(PROJECT_SOURCES
#Place .cpp files here
)

set(PROJECT_HEADERS
#Place .h files here
)

# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_CUDA})
# Compile the CUDA files separately. They can still be linked.
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)