/* Hello GLFW
*  Based on https://www.glfw.org/docs/3.3/quick.html
*/

#include "happly/happly.h"
#include "main.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
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

GLuint loadShaders(const char* vertex_file_path, const char* fragment_file_path) {

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
	if (VertexShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	}
	else {
		printf("Impossible to open %s. Are you in the right directory?\n", vertex_file_path);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
	if (FragmentShaderStream.is_open()) {
		std::stringstream sstr;
		sstr << FragmentShaderStream.rdbuf();
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertex_file_path);
	char const* VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragment_file_path);
	char const* FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer, NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}

	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);

	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}


int main()
{

	const int w = 1080, h = 1080;

	//Initialize GLFW
	if (!glfwInit())
	{
		cerr << "GLFW initialization failed :(";
		terminate(1);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwSetErrorCallback(error_callback);
	GLFWwindow* window = glfwCreateWindow(w, h, "Hello window :)", NULL, NULL);
	if (!window)
	{
		cerr << "GLFW failed to create a window and/or OpenGL context :(";
		terminate(1);
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);


	//  Initialize glad (must occur AFTER window creation or glad will error)
	if (!gladLoadGL()) {
		std::cerr << "Failed to initialize OpenGL context" << std::endl;
	}
	glViewport(0, 0, w, h); // use a screen size of WIDTH x HEIGHT
	glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
	glLoadIdentity();
	glOrtho(0.0, w, h, 0.0, 0.0, 100.0);
	glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling
	glClearColor(0.f, 0.f, 0.f, 0.f); //Set the clear color
	glClearDepth(100.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	//Initialize shaders
	GLuint shaderID = loadShaders(MEDIA_DIRECTORY "shaders/pointcloud.vert", MEDIA_DIRECTORY "shaders/passthrough.frag");
	GLuint viewMatrixID = glGetUniformLocation(shaderID, "ViewMatrix");
	GLuint transformMatrixID = glGetUniformLocation(shaderID, "TransformMatrix");

	// Make a vertex array
	GLuint vertexArray;
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);
	// Construct the data object by reading from file
	happly::PLYData plyIn(MEDIA_DIRECTORY "german-shepherd-pointcloud.ply");
	// Get mesh-style data from the object
	std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
	std::vector<std::array<unsigned char, 3>> vCol = plyIn.getVertexColors();
	// Make a VBO for the positions
	GLuint posBufferID, colBufferID;
	std::vector<GLfloat> posBuffer (vPos.size() * 3);
	std::vector<GLfloat> colBuffer (vPos.size() * 3);
	for (size_t i = 0; i < vPos.size(); i++) {
		posBuffer[i * 3 + 0] = (GLfloat)vPos[i][0];
		posBuffer[i * 3 + 1] = (GLfloat)vPos[i][1];
		posBuffer[i * 3 + 2] = (GLfloat)vPos[i][2];
		colBuffer[i * 3 + 0] = (GLfloat)vCol[i][0] / 256.0;
		colBuffer[i * 3 + 1] = (GLfloat)vCol[i][1] / 256.0;
		colBuffer[i * 3 + 2] = (GLfloat)vCol[i][2] / 256.0;
	}
	glGenBuffers(1, &posBufferID);
	glGenBuffers(1, &colBufferID);




	//The render loop
	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
		glUseProgram(shaderID);
		glBindBuffer(GL_ARRAY_BUFFER, posBufferID);
		glBufferData(GL_ARRAY_BUFFER, posBuffer.size() * sizeof(GLfloat), posBuffer.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, colBufferID);
		glBufferData(GL_ARRAY_BUFFER, colBuffer.size() * sizeof(GLfloat), colBuffer.data(), GL_STATIC_DRAW);
		//Uniforms
		float cameraPosition[] = { -2, 1, 2};
		float cameraDirection[] = { 1,-.5,-1 };
		//Recalculate the view matrix
		glm::vec3 _eye(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
		glm::vec3 _center(cameraPosition[0] + cameraDirection[0],
			cameraPosition[1] + cameraDirection[1],
			cameraPosition[2] + cameraDirection[2]);
		glm::vec3 _up(0, 1, 0);
		glm::mat4 viewMatrix = glm::lookAt(_eye, _center, _up);
		viewMatrix = glm::perspective(glm::radians(45.0f), 1.f, 0.1f, 10.f) * viewMatrix; // Assumes a square (1:1) aspect ratio
		//Calculate the projection matrix.
		glm::mat4 transformMatrix = glm::mat4(1.0);
		//Apply the matrix
		glUniformMatrix4fv(viewMatrixID, 1, GL_FALSE, &viewMatrix[0][0]);
		glUniformMatrix4fv(transformMatrixID, 1, GL_FALSE, &transformMatrix[0][0]);

		// 1st attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, posBufferID);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		// 2nd attribute buffer : colors
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, colBufferID);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		// Draw the triangle!
		glDrawArrays(GL_POINTS, 0, posBuffer.size()); // 3 index starting at 0

		// End the drawing process
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);


		glfwSwapBuffers(window);
		glfwPollEvents();
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}

	//Close GLFW
	glfwTerminate();
}

