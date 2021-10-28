#include "renderer.hpp"
#include "glm/gtc/matrix_transform.hpp"


using namespace PointCloud;


GLuint Renderer::shaderID;
GLuint Renderer::uniformViewID, Renderer::uniformTransformID;
GLuint Renderer::vertexArrayID, Renderer::positionBufferID, Renderer::colorBufferID;
bool Renderer::initialized = false;
glm::mat4 Renderer::viewMatrix;


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


void Renderer::set(Data<6> xyzrgb) {
	initialize();

	std::vector<GLfloat> xyz (xyzrgb.size()*3);
	std::vector<GLfloat> rgb (xyzrgb.size()*3);
	for (size_t i = 0; i < xyzrgb.size(); i++) {
		xyz[i * 3 + 0] = xyzrgb[i][0];
		xyz[i * 3 + 1] = xyzrgb[i][1];
		xyz[i * 3 + 2] = xyzrgb[i][2];
		rgb[i * 3 + 0] = xyzrgb[i][3];
		rgb[i * 3 + 1] = xyzrgb[i][4];
		rgb[i * 3 + 2] = xyzrgb[i][5];
	}

	glBindBuffer(GL_ARRAY_BUFFER, positionBufferID);
	glBufferData(GL_ARRAY_BUFFER, xyz.size() * sizeof(GLfloat), xyz.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
	glBufferData(GL_ARRAY_BUFFER, rgb.size() * sizeof(GLfloat), rgb.data(), GL_STATIC_DRAW);

	size = xyz.size();
}

void PointCloud::Renderer::setSparse(Data<6> xyzrgb, double percent)
{
	size = (size_t)(xyzrgb.size() * percent);
	size_t incr = xyzrgb.size() / size;
	size -= 1;

	initialize();

	std::vector<GLfloat> xyz (size * 3);
	std::vector<GLfloat> rgb (size * 3);
	for (size_t i = 0; i < size; i ++) {
		xyz[i * 3 + 0] = xyzrgb[i*incr][0];
		xyz[i * 3 + 1] = xyzrgb[i*incr][1];
		xyz[i * 3 + 2] = xyzrgb[i*incr][2];
		rgb[i * 3 + 0] = xyzrgb[i*incr][3];
		rgb[i * 3 + 1] = xyzrgb[i*incr][4];
		rgb[i * 3 + 2] = xyzrgb[i*incr][5];
	}

	glBindBuffer(GL_ARRAY_BUFFER, positionBufferID);
	glBufferData(GL_ARRAY_BUFFER, xyz.size() * sizeof(GLfloat), xyz.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
	glBufferData(GL_ARRAY_BUFFER, rgb.size() * sizeof(GLfloat), rgb.data(), GL_STATIC_DRAW);

	size = xyz.size();
}

void PointCloud::Renderer::display()
{
	// Configuration
	glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling
	glClearColor(0.f, 0.f, 0.f, 0.f); //Set the clear color
	glClearDepth(100.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	//glDisable(GL_DEPTH_TEST);

	// Shading
	glUseProgram(shaderID);
	//Uniforms
	glUniformMatrix4fv(uniformViewID, 1, GL_FALSE, &viewMatrix[0][0]);
	glUniformMatrix4fv(uniformTransformID, 1, GL_FALSE, &transformMatrix[0][0]);

	// 1st attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, positionBufferID);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// 2nd attribute buffer : colors
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// Draw the triangle!
	glDrawArrays(GL_POINTS, 0, size); // 3 index starting at 0

	// End the drawing process
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}

void PointCloud::Renderer::setViewLookAt(float cameraPosition[3], float cameraCenter[3])
{
	glm::vec3 _eye(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	glm::vec3 _center(cameraCenter[0], cameraCenter[1], cameraCenter[2]);
	glm::vec3 _up(0, 1, 0);

	glm::mat4 v = glm::perspective(glm::radians(45.0f), 1.f, 0.1f, 10.f) * glm::lookAt(_eye, _center, _up);
	glm::mat4 t = glm::mat4(1.0);

	viewMatrix = v;
	transformMatrix = t;
}


void PointCloud::Renderer::initialize()
{
	if (initialized) return;
	initialized = true;

	//Initialize shaders
	shaderID = loadShaders(MEDIA_DIRECTORY "shaders/pointcloud.vert", MEDIA_DIRECTORY "shaders/passthrough.frag");
	uniformViewID = glGetUniformLocation(shaderID, "ViewMatrix");
	uniformTransformID = glGetUniformLocation(shaderID, "TransformMatrix");

	// Make a vertex array
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	// Make a VBO for the positions
	glGenBuffers(1, &positionBufferID);
	glGenBuffers(1, &colorBufferID);
}