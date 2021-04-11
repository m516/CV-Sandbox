# CMakeList.txt : CMake project for CV-Sandbox, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# This is the name of the executable
set(EXECUTABLE_NAME 11_GL_GAME)

# In UNIX, set PROJECT_SOURCES using the following command
# find . | grep .cpp
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_SOURCES
"behavior/gameObject.cpp"
"behavior/gameObjects/test_triangle.cpp"
"behavior/gameObjects/test_cube.cpp"
"behavior/gameObjects/objFileObject.cpp"
"behavior/world.cpp"
"engine/renderer.cpp"
"engine/shader.cpp"
"main.cpp"
"utils.cpp"
)

# In UNIX, set PROJECT_HEADERS using the following command
# find . | grep .h
# Delete the ./ at the beginning of each file with block selection (alt + shift in Visual Studio)
set(PROJECT_HEADERS
"behavior/gameObject.h"
"behavior/gameObjects/test_triangle.h"
"behavior/gameObjects/test_cube.h"
"behavior/gameObjects/objFileObject.h"
"behavior/world.h"
"behavior/worlds/w1_triangle.h"
"behavior/worlds/w2_cube.h"
"behavior/worlds/w3_objects.h"
"engine/renderer.h"
"engine/shader.h"
"engine/stb_image.h"
"main.h"
"utils.h"
)


# Add source to this project's executable.
add_executable (${EXECUTABLE_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})

# Add tests and install targets if needed.
target_link_libraries (${EXECUTABLE_NAME} glad glfw) 