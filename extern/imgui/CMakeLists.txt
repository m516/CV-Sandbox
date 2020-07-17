cmake_minimum_required(VERSION 3.6)

#find_package(GLFW3 CONFIG REQUIRED)
#include_directories(../glfw)
#add_subdirectory(../glfw glfw)

add_compile_definitions(IMGUI_IMPL_OPENGL_LOADER_GLAD=1)

add_library(
        imgui STATIC
        imgui.cpp
        imgui_draw.cpp
        imgui_internal.h
        imstb_textedit.h
        imgui.h
        imgui_impl_glfw.cpp
        imgui_widgets.cpp
        imstb_truetype.h
        imconfig.h
        imgui_demo.cpp
        imgui_impl_glfw.h
        imstb_rectpack.h
        imgui_impl_opengl3.cpp
        imgui_impl_opengl3.h
 "imgui_impl_opengl3.cpp" "imgui_impl_opengl3.h")

target_link_libraries(imgui PRIVATE glfw glad)