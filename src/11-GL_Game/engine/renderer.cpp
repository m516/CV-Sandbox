#include "renderer.h"
#include "shader.h"



void Renderer::init(World *world){
	using namespace std::chrono;
	//Attempt to initialize GLFW
	if (!glfwInit())
	{
		//Initialization failed
		RUNTIME_ERROR("GLFW initialization failed :(");
	}

	//Set the requirements for the version of OpenGL to use
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); //Reqired for Mac OS X

	//Set GLFW's error callback function
	glfwSetErrorCallback(error_callback);

	//GLFW creates a window and its OpenGL context with the next function
	_window = glfwCreateWindow(640, 480, "Simple Graphics Processor Game Engine", NULL, NULL);

	//Check for errors (which would happen if creating a window fails
	if (!_window)
	{
		// Window or OpenGL context creation failed
		RUNTIME_ERROR("GLFW failed to create a window and/or OpenGL context :(");
	}

	//Window creation was successful. Continue
	glfwMakeContextCurrent(_window);

	//  Initialise glad (must occur AFTER window creation or glad will error)
    if (!gladLoadGL()) {
        // GLAD failed
        RUNTIME_ERROR("GLAD failed to initialize :(");
    }

	//Set the clear color
	glClearColor(0.1f, 0.1f, 0.14f, 0.1f);

	//Shaders
	_shader = world->_shader;
	_shader->_init();
}

void Renderer::render(){
    using namespace std;
	using namespace std::chrono;


	//The render loop
	if (glfwWindowShouldClose(_window)){
		close();
	}
	else{
		//TODO Shader
		_shader->_apply();

		//Allocate the OpenGL memory buffers
		if(!vertexArray._boundToGL()) vertexArray._allocateBuffers();
		vertexArray._sync();

		glBindBuffer(GL_ARRAY_BUFFER, vertexArray._vertexPositionBuffer);
		glBufferData(GL_ARRAY_BUFFER, vertexArray.size()*3*sizeof(GLfloat), vertexArray._vertexPositionBufferData, GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, vertexArray._vertexColorBuffer);
		glBufferData(GL_ARRAY_BUFFER, vertexArray.size()*3*sizeof(GLfloat), vertexArray._vertexColorBufferData, GL_STATIC_DRAW);

		// 1st attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexArray._vertexPositionBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// 2nd attribute buffer : colors
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, vertexArray._vertexColorBuffer);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// Draw the triangle!
		glDrawArrays(GL_TRIANGLES, 0, vertexArray.size()); // 3 index starting at 0

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

    	glfwPollEvents();
    	glfwSwapBuffers(_window);
		//this_thread::sleep_for(milliseconds(16));
	}
}

void Renderer::close(){
	//Close GLFW
	glfwTerminate();
	std::cout << "Closing the application normally." << std::endl;
}