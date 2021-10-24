# This macro finds all the libraries required by the projects. 
# 
# It has one parameter: the path to the folder containing all the 
# libraries that are stored locally in this project.

cmake_minimum_required(VERSION 3.6)


macro (get_3rd_party_libraries)

###########################################################################################################
#                                         Argument Checking                                               #
###########################################################################################################
# The first parameter is the path to the external library, relative to where the macro was called
if(${ARGC} GREATER 0)
    set(EXTERNAL_LIBRARY_DIRECTORY ${ARGV0}) # used by external_dependencies to locate the external packages
else()
    message("${BoldRed}Error from get_external_dependencies macro: External library directory not set!${ColorReset}")
endif()

###########################################################################################################
#                                         External Libraries                                              #
###########################################################################################################
message("${BoldMagenta}Searching for external libraries...${ColorReset}")

# OpenCV
message("${BoldYellow}Finding OpenCV${ColorReset}")
option(USE_OPENCV "Attempt to use OpenCV. If disabled, projects that require OpenCV can't build. Enabled by default" ON)
if(USE_OPENCV)
    find_package(OpenCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    if(OpenCV_FOUND)
        message("${BoldYellow}OpenCV found.${ColorReset}")
    else()
        message("${BoldRed}OpenCV wasn't found.${ColorReset}")
    endif()

else()
    message("${BoldRed}OpenCV was disabled.${ColorReset}")
endif()

###########################################################################################################
#                                         Internal Libraries                                              #
###########################################################################################################
message("${BoldMagenta}Importing local libraries...${ColorReset}")
include_directories(${EXTERNAL_LIBRARY_DIRECTORY})


# glad
message("${BoldYellow}Using local copy of GLAD${ColorReset}")
add_subdirectory("${EXTERNAL_LIBRARY_DIRECTORY}/glad")

# glfw
message("${BoldYellow}Using local copy of GLFW${ColorReset}")
add_subdirectory("${EXTERNAL_LIBRARY_DIRECTORY}/glfw")

# Dear ImGui
message("${BoldYellow}Using local copy of ImGui${ColorReset}")
add_subdirectory("${EXTERNAL_LIBRARY_DIRECTORY}/imgui")

# glm
message("${BoldYellow}Using local copy of GLM${ColorReset}")
add_subdirectory("${EXTERNAL_LIBRARY_DIRECTORY}/glm")
include_directories("${EXTERNAL_LIBRARY_DIRECTORY}/glm")

# hapPLY
message("${BoldYellow}Using local copy of hapPLY${ColorReset}")
# Nothing to do. This is a header-only directory

# That's all the libraries
message("${BoldYellow}Library search complete!${ColorReset}")

endmacro()