#include "utils.h"


void terminate(int errorCode) {
    using namespace std;
	cout << "Closing application" << endl;
	//Close GLFW
	glfwTerminate();
	//Exit
	exit(errorCode);
}

void error_callback(int error, const char* description)
{
    //Use this function to safely exit the program
    RUNTIME_ERROR("Error: %s\n", description);
    //Use this function to print the error and attempt to continue
	//fprintf(stderr, "Error: %s\n", description);
}