# CMakeLists.txt

# This is the name of the executable
set(EXECUTABLE_NAME 10_CONVOLUTION_BENCHMARK)


include(CheckIncludeFiles)
check_include_file(unistd.h HAVE_UNISTD_H HAVE_UNISTD_H)


set(PROJECT_CUDA
)

set(PROJECT_SOURCES
main.c
conv4D_data_structures.c
conv4D_impl_CPU.c
)

set(PROJECT_HEADERS
conv4D_impl.h
conv4D_data_structures.h
)


#CUDA
if(CMAKE_CUDA_COMPILER)
    message("${Green}             CUDA found${ColorReset}")
    #Add compiler definition
    if(CMAKE_VERSION VERSION_GREATER 3.12)
        add_compile_definitions(CUDA_SUPPORT)
    else()
        add_definitions("-DCUDA_SUPPORT")
    endif()
    #Add source
    set(PROJECT_CUDA
        ${PROJECT_CUDA}
        conv4D_impl_GPU.cu
    )
else()
    message("${Red}             CUDA not found${ColorReset}")
endif()
#OpenMP
if(OpenMP_FOUND)
    message("${Green}             OpenMP found${ColorReset}")
    if(CMAKE_VERSION VERSION_GREATER 3.12)
        add_compile_definitions(OMP_SUPPORT)
    else()
        add_definitions("-DOMP_SUPPORT")
    endif()
else()
    message("${Red}             OpenMP not found${ColorReset}")
endif()
#Threads
if(CMAKE_USE_PTHREADS_INIT)
    message("${Green}             Threads found${ColorReset}")
    #THERAD_SUPPORT already defined at root of project
else()
    message("${Red}             Threads not found${ColorReset}")
endif()






# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_CUDA})
# Compile the CUDA files separately. They can still be linked.
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# Link to libraries
# Threads
target_link_libraries(${EXECUTABLE_NAME} Threads::Threads)
target_link_libraries(${EXECUTABLE_NAME} OpenMP::OpenMP_CXX)