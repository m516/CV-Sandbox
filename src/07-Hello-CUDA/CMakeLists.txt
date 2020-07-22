# CMakeList.txt

# This is the name of the executable
set(EXECUTABLE_NAME 07_HELLO_CUDA)

if(CMAKE_CUDA_COMPILER)
    
    set(PROJECT_CUDA
    HelloCUDA.cu
    )

    set(PROJECT_SOURCES
    #HelloCUDA.cpp
    )

    set(PROJECT_HEADERS
    HelloCUDA.h
    )

    # Add source to this project's executable.
    add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_CUDA})

    set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

else()
    message("${BoldRed}    Project can't be built without CUDA${ColorReset}")

endif()