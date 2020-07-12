# Project 3: Hello ImGui <!-- omit in toc -->
This project demonstrates the use of GLFW to create a window with an OpenGL context in a [single C++ source code file.](main.cpp)

# Table of Contents <!-- omit in toc -->
- [The Gist](#the-gist)
	- [GLAD disabled for now](#glad-disabled-for-now)
	- [Changes to the `main()` function](#changes-to-the-main-function)
		- [ImGui is initialized](#imgui-is-initialized)
		- [Render loop: building ImGui widgets](#render-loop-building-imgui-widgets)
		- [Render loop: rendering ImGui widgets](#render-loop-rendering-imgui-widgets)
- [Resources](#resources)

# The Gist
This project is similar in structure to Project 2: Hello GLFW. The most significant difference between the two projects is that this one employs ImGui to build and render a simple GUI on top of the window created by GLFW.

The following changes were made to use ImGui:

## GLAD disabled for now
GLAD caused conflicts with GLEW, so it has been removed for now, as ImGui requires GLEW.

## Changes to the `main()` function

### ImGui is initialized
After the window has been created, but before the first render loop, the following code was written to initialize ImGui and to bind ImGui to the
GLFW window.

```C++
// Setup ImGui binding
ImGui_ImplGlfwGL3_Init(window, true);
setStyle();
```

### Render loop: building ImGui widgets
Under the main render loop is a series of commands that tell ImGui about the
structure of the GUI I want to build, first by creating a window that's 320 pixels
wide and 240 pixels tall, adding a label that says "hello" to that window,
and creating a button under that label. If the button is clicked, a new label is created under the button.
```C++
ImGui::SetNextWindowSize(ImVec2(320,240));
ImGui::Begin("Another Window", &show_another_window);
ImGui::Text("Hello");
if (ImGui::Button("Push me", ImVec2(128, 32))) {
	ImGui::Text("Ouch, not so hard!");
}
ImGui::End();
```


### Render loop: rendering ImGui widgets
The GLFW port for ImGui contains functions that allow ImGui to be rendered
to the GLFW window.
```C++
// Rendering
int display_w, display_h;
glfwGetFramebufferSize(window, &display_w, &display_h);
glViewport(0, 0, display_w, display_h);
glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
glClear(GL_COLOR_BUFFER_BIT);
ImGui::Render();
```



# Resources
* [ImGui Github repository](https://github.com/ocornut/imgui)
* [GLFW monitor handling (for high-DPI scaling)](https://www.glfw.org/docs/latest/monitor.html)
* [My fork of an unofficial demo](https://github.com/m516/imgui-opengl-glfw-glew-cmake-demo/)
