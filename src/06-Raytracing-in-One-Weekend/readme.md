![Current screenshot](../../docs/screenshots/08-01.png)

# Project 8: OpenCV Video Template <!-- omit in toc -->
This project opens a video, plays it, and exposes an OpenCV Mat instance
and simple API to control the video feed and render the current Mat in real time
on a GUI. It uses the GUI setup from Project 6 on the tools from Project 5.

## Table of Contents <!-- omit in toc -->
- [Usage](#usage)
  - [Video Viewer](#video-viewer)
  - [Window Settings](#window-settings)
  - [Style Editor](#style-editor)
  - [Demo Window](#demo-window)
- [Framework](#framework)
- [Resources](#resources)
  - [Code samples](#code-samples)
  - [Documentation](#documentation)


## Usage

Four windows allow the user to control many aspects of the application:

### Video Viewer
This window shows the rendered image on the screen and some useful information about the image and its underlying OpenGL texture. These are its controls:

1. **Video Position** allows users to view the index if the current frame and to scrub the video feed.
2. **Loop** starts the video at the beginning if the next frame couldn't be displayed.
3. **Play** advances to the next frame every time the screen refreshes.
4. **Next frame** advances to the next frame.
5. **Display Scale** allows users to zoom in to and out of the image.

Hovering over the image creates a widget that allows the user to see a 4x magnified slice of the image.


### Window Settings
This window controls the graphics settings:
1. **Display Scale** sets the magnification of the GUI, which may increase the viewing experience of users with high-DPI displays. This defaults to the operating system's internal scale, but it can be set to any number.
2. **Swap Interval** sets the number of monitor-refreshes to wait until the GUI redraws.

### Style Editor
This allows the user to customize the aesthetics of the GUI. It can be exported to C++ code.

It is a window provided by and included in the ImGui source code.

### Demo Window
This is a demo window provided by and included in the ImGui source code.

## Framework
Four main file pairs control the user interface
* `main.cpp`/`main.h`: a shell for starting the application. Only contains `gui::begin();` in a `main()` function.
* `app.cpp`/`app.h`: responsible for creating a window with GLFW, loading OpenGL through GLAD, maintaining a render loop, initializing ImGui before rendering, and rendering the GUI at the end of the render loop. It also contains some application-specific helper functions in the namespace `gui`:
  * `float estimateSystemScale()`: uses GLFW to estimate the operating system's native scale.
  * `GLFWwindow* getCurrentWindow()`: gets the current GLFWwindow instance.
  * `void setSwapInterval(int newSwapInterval)`: sets the window's swap interval, i.e. how many frame refreshes before the GUI refreshes. This also sets the UI's framerate. Higher swap interval values yield a lower framerate, but reduce the CPU consumption and increase battery life of machines running the software. Note that this is specific to the operating system and may not work effectively.
  * `void setClearColor(int red, int green, int blue, int alpha)`: sets the color of the background.
  * `void setGuiScale(float guiScale)`: sets the scale of the GUI.
  * Most importantly `gui::begin()` starts the GUI and is called from `main`. 
* `ui.cpp`/`ui.h`: responsible for the GUI layout, controls, and behavior of the application. Without this files, only a blank window will be displayed.
  * `void setStyle()`: sets the look-and-feel of the UI. The settings I provide in this code create a gray background with black windows, white text, and orange highlights, but all the preferences I set can be changed in the style editor built into the application.
  * `void populateUI()`: responsible for the layout of the entire user interface.
  * `void initUI()`: initializes the variables and settings used for the user interface prior to the first time `void populateUI()` is called.


Therefore, `populateUI()`is not at all responsible for rendering the UI, yet 
it controls everything about the struture, layout, and appearance of the UI. Likewise,
`render()` controls the window and all tools used to display the GUI, but doesn't need to 
worry about how the GUI is set up or what it does.


## Resources
### Code samples
* [Project 05: OpenCV and ImGui](../05-OpenCV-and-ImGui)
* [Project 06: Raytracing in One Weekend](../06-Raytracing-in-One-Weekend)
### Documentation
