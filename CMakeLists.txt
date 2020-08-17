# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.10)

# Define colors
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(ColorBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

project ("CV-Sandbox")

#Manually place all binaries into a folder under bin/
set(BUILD_FULL_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin/)
message("${BoldMagenta}Output directory: ${BUILD_FULL_DIRECTORY}${ColorReset}")
# First for the generic no-config case (e.g. with mingw)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${BUILD_FULL_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${BUILD_FULL_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${BUILD_FULL_DIRECTORY})
# Second, for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BUILD_FULL_DIRECTORY})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BUILD_FULL_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${BUILD_FULL_DIRECTORY})
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

# Manually define a compile-time definition for the media/ directory
set(MEDIA_DIR "${CMAKE_SOURCE_DIR}/media/")
if(CMAKE_VERSION VERSION_GREATER 3.12)
    add_compile_definitions(MEDIA_DIRECTORY="${MEDIA_DIR}")
else()
    add_definitions("-DMEDIA_DIRECTORY=\"${MEDIA_DIR}\"")
endif()
message("${BoldMagenta}Media directory: ${MEDIA_DIR}${ColorReset}")

# Set the macro/helper directory 
set(MACRO_DIRECTORY ${CMAKE_HOME_DIRECTORY}/CMake/macros)
message("${BoldMagenta}CMake utility directory: ${MACRO_DIRECTORY}${ColorReset}")
# Add some macros
include(${MACRO_DIRECTORY}/libhelpers.cmake)

# Find packages used throughout this repository

message("${BoldYellow}Searching for libraries...${ColorReset}")

# OpenCV
message("${BoldYellow}Finding OpenCV${ColorReset}")
option(USE_OPENCV "Attempt to use OpenCV. If disaled, projects that require OpenCV can't build. Enabled by default" ON)
if(USE_OPENCV)
    find_package(OpenCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message("${BoldYellow}OpenCV found at: ${OpenCV_INCLUDE_DIRS}${ColorReset}")
else()
    message("${BoldRed}OpenCV was disabled.${ColorReset}")
endif()

# glad
message("${BoldYellow}Using local copy of GLAD${ColorReset}")
add_subdirectory(extern/glad)
include_directories(extern/glad)

# glfw
message("${BoldYellow}Using local copy of GLFW${ColorReset}")
add_subdirectory(extern/glfw)

# Dear ImGui
message("${BoldYellow}Using local copy of ImGui${ColorReset}")
add_subdirectory(extern/imgui)
include_directories(extern)

message("${BoldYellow}Library search complete!${ColorReset}")

#CUDA
message("${BoldYellow}Checking if CUDA files can be compiled${ColorReset}")
option(USE_CUDA "Attempt to use CUDA. If disaled, projects that require CUDA can't build. Enabled by default" ON)
#https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message("${BoldYellow}CUDA exists${ColorReset}")
        enable_language(CUDA)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()
    else()
        message("${BoldRed}Failed to find CUDA. Some projects can't be run.${ColorReset}")
    endif()
else()
    message("${BoldRed}CUDA was disabled.${ColorReset}")
endif()
    

# Include sub-projects.
add_subdirectory ("src")
