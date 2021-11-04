#include "window.hpp"
#include <iostream>
#include <vector>

std::vector<App::Window*> global_appWindowList(1);



/**
 * A callback function for GLFW to execute when an internal error occurs with the
 * library.
 */
void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void resize_callback(GLFWwindow* window, int new_width, int new_height) {
	glViewport(0, 0, new_width, new_height);
	for (int i = 0; i < global_appWindowList.size(); i++) {
		if (global_appWindowList[i] == nullptr) continue;
		if (global_appWindowList[i]->window() == window) global_appWindowList[i]->resize(new_height, new_height, false);
	}
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
		_window = glfwCreateWindow(width, height, "CV-Sandbox", NULL, NULL);
		if (!_window)
		{
			cerr << "GLFW failed to create a window and/or OpenGL context :(";
			return;
		}
		glfwSetWindowSizeCallback(_window, resize_callback);
		glfwMakeContextCurrent(_window);
		glfwSwapInterval(1);
		//  Initialize glad (must occur AFTER window creation or glad will error)
		if (!gladLoadGL()) {
			std::cerr << "Failed to initialize OpenGL context" << std::endl;
		}
		glViewport(0, 0, width, height); // use a screen size of WIDTH x HEIGHT


		_open = true;

		global_appWindowList.push_back(this);
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

	void Window::resize(int width, int height, bool doResize)
	{
		if (doResize)
			glfwSetWindowSize(_window, width, height);

		glfwMakeContextCurrent(_window);
		glViewport(0, 0, width, height);
	}

	void Window::getDimensions(int& w, int& h)
	{
		glfwGetWindowSize(_window, &w, &h);
	}

	int Window::getWidth()
	{
		int w, h;
		glfwGetWindowSize(_window, &w, &h);
		return w;
	}

	int Window::getHeight()
	{
		int w, h;
		glfwGetWindowSize(_window, &w, &h);
		return h;
	}

	float Window::getAspectRatio()
	{
		int w, h;
		glfwGetWindowSize(_window, &w, &h);
		return (float)(w) / (float)(h);
	}

}

