/* Hello GLFW
*  Based on https://www.glfw.org/docs/3.3/quick.html
*/

#include "main.h"
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
