#include "window.hpp"
#include <iostream>



/**
 * A callback function for GLFW to execute when an internal error occurs with the
 * library.
 */
void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}


namespace App {
	Window::Window()
	{
	}

	Window::Window(int width, int height)
	{
		using namespace std;

		//Initialize GLFW
		if (!glfwInit())
		{
			cerr << "GLFW initialization failed :(";
			return;
		}
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwSetErrorCallback(error_callback);
		_window = glfwCreateWindow(width, height, "Hello window :)", NULL, NULL);
		if (!_window)
		{
			cerr << "GLFW failed to create a window and/or OpenGL context :(";
			return;
		}
		glfwMakeContextCurrent(_window);
		glfwSwapInterval(1);
		//  Initialize glad (must occur AFTER window creation or glad will error)
		if (!gladLoadGL()) {
			std::cerr << "Failed to initialize OpenGL context" << std::endl;
		}
		glViewport(0, 0, width, height); // use a screen size of WIDTH x HEIGHT


		_open = true;
	}

	bool Window::shouldClose()
	{
		return glfwWindowShouldClose(_window);
	}

	void Window::refresh()
	{
		glfwSwapBuffers(_window);
		glfwPollEvents();
	}

	void Window::close()
	{
		//Close GLFW
		glfwTerminate();
	}

	void Window::clear()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
	}

}

