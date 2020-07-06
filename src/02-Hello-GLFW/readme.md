# Project 2: Hello GLFW
This project demonstrates the use of GLFW to create a window with an OpenGL context in a [single C++ source code file.](main.cpp)

## The Gist
This is the contents of the main.cpp file as of 7/3/2020
```C++
#include "main.h"
#include <glad/glad.h> // ALWAYS place this BEFORE GLFW in the list of headers to include! This REPLACES OpenGL
#include <GLFW/glfw3.h>
#include <thread>

using namespace std;



/**
 * A helper function for terminating the program
 */
void terminate(int errorCode) {
	cout << "Closing application";
	//Close GLFW
	glfwTerminate();
	//Exit
	exit(errorCode);
}


/**
 * A callback function for GLFW to execute when an internal error occurs with the
 * library.
 */
void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}


int main()
{
	//Attempt to initialize GLFW
	if (!glfwInit())
	{
		//Initialization failed
		cerr << "GLFW initialization failed :(";
		//Use the terminate() function to safely close the application
		terminate(1);
	}

	//Set the requirements for the version of OpenGL to use
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); //Reqired for Mac OS X

	//Set GLFW's error callback function
	glfwSetErrorCallback(error_callback);

	//GLFW creates a window and its OpenGL context with the next function
	GLFWwindow* window = glfwCreateWindow(640, 480, "Hello window :)", NULL, NULL);

	//Check for errors (which would happen if creating a window fails
	if (!window)
	{
		// Window or OpenGL context creation failed
		cerr << "GLFW failed to create a window and/or OpenGL context :(";
		//Use the terminate() function to safely close the application
		terminate(1);
	}

	//Window creation was successful. Continue
	glfwMakeContextCurrent(window);

	//Initialize GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//Tell OpenGL the size of the rendering window, so it knows where we want
	//to display stuff
	glViewport(0, 0, 640, 480);

	//The render loop
	while (!glfwWindowShouldClose(window))
	{
		glfwSwapBuffers(window);
		glfwPollEvents();
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}

	//Close GLFW
	glfwTerminate();
}
```


## Resources
### For Building
* [Official GLFW build guide](https://www.glfw.org/docs/latest/build_guide.html)

### Examples
* [Official GLFW quick start guide](https://www.glfw.org/docs/latest/quick.html)
* [imgui + GLFW + CMake demo](https://github.com/m516/imgui-opengl-glfw-glew-cmake-demo/)
* LearnOpenGL.com
	* [Hello Window](https://learnopengl.com/Getting-started/Hello-Window)
	* [Creating a Window](https://learnopengl.com/Getting-started/Creating-a-window)

### Miscellaneous
* [GLAD Github repository](https://github.com/Dav1dde/glad)
