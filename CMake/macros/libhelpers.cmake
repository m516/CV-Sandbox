macro (disable_if_cuda_not_found)
    #Check if CUDA is unavailable
    if(NOT CMAKE_CUDA_COMPILER)
        message("${BoldRed}             Project can't be built without CUDA${ColorReset}")
        return()
    endif()
endmacro()

macro (disable_if_opencv_not_found)
    #Check if CUDA is unavailable
    if(NOT OpenCV_FOUND)
        message("${BoldRed}             Project can't be built without OpenCV${ColorReset}")
        return()
    endif()
endmacro()

macro (disable_if_pthread_not_found)
    #Check if CUDA is unavailable
    if(NOT Threads_FOUND)
        message("${BoldRed}             Project can't be built without Pthreads${ColorReset}")
        return()
    endif()
endmacro()
