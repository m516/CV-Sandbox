## Computer Vision Sandbox

All my personal, non-academic computer vision research lies here.

So far, all my projects are written in C++. This allows me to practice using standard, free tools to make fast, lightweight standalone applications that are relatively easy to integrate with hardware like GPUs and TPUs. Check out my setup at [docs/setup.md](docs/setup.md)

Here are all my projects so far:

|                                      Name | Description                                                                                                         |
| ----------------------------------------: | :------------------------------------------------------------------------------------------------------------------ |
|         [Hello World](src/01-Hello-World) | A basic project that loads with OpenCV and displays it in a window                                                  |
|           [Hello GLFW](src/02-Hello-GLFW) | A basic project that creates a window and an OpenGL context with GLFW and GLAD                                      |
|         [Hello ImGui](src/03-Hello-ImGui) | A basic project that uses the GLFW window created from project 02 to create a simple Gui with ImGui                 |
| [GLFW and OpenCV](src/04-GLFW-and-OpenCV) | A combination of Projects 01 and 02, loading an image with OpenCV and displaying the generated OpenCV Mat with GLFW |
| [GLFW and OpenCV and ImGui](src/05-OpenCV-and-ImGui) | A combination of Projects 03 and 04, loading an image with OpenCV and displaying the generated OpenCV Mat with GLFW *on an ImGui window* |
