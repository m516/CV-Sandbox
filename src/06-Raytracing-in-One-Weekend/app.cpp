#include "app.h"

using namespace std;

namespace gui {

	bool initialized = false;
	GLFWwindow* currentGLFWWindow = nullptr;
	int window_width = 0, window_height = 0;
	ImVec4 clear_color = ImColor(96, 96, 96);



	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}
	}

	static void resize_callback(GLFWwindow* window, int new_width, int new_height) {
		glViewport(0, 0, window_width = new_width, window_height = new_height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, window_width, window_height, 0.0, 0.0, 100.0);
		glMatrixMode(GL_MODELVIEW);
	}

	/**
	 * A callback function for GLFW to execute when an internal error occurs with the
	 * library.
	 */
	void error_callback(int error, const char* description)
	{
		fprintf(stderr, "Error: %s\n", description);
	}

	void destroy() {
		cout << "Closing application";

		// Close ImGui
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		// Close GLFW
		if (currentGLFWWindow) glfwDestroyWindow(currentGLFWWindow);
		glfwTerminate();
	}

	void setup() {
		// Setup window
		glfwSetErrorCallback(error_callback);
		if (!glfwInit())
			return;

		// Decide GL+GLSL versions
#if __APPLE__
	// GL 3.2 + GLSL 150
		const char* glsl_version = "#version 150";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
		const char* glsl_version = "#version 130";
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
		//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
		//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

	// Create window with graphics context
		currentGLFWWindow = glfwCreateWindow(1280, 720, "Raytracing in One Weekend", NULL, NULL);
		if (currentGLFWWindow == NULL)
			return;
		glfwMakeContextCurrent(currentGLFWWindow);
		glfwSwapInterval(3); // Enable vsync

		if (!gladLoadGL()) {
			// GLAD failed
			cerr << "GLAD failed to initialize :(";
			//Use the terminate() function to safely close the application
			destroy();
			return;
		}

		//Sets the number of screen updates to wait before swapping the buffers of a window
		//Handles vertical synchronization
		setSwapInterval(1);

		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui_ImplGlfw_InitForOpenGL(currentGLFWWindow, true);
		ImGui_ImplOpenGL3_Init(glsl_version);
		setStyle();

		//Initialize UI
		initUI();
	}

	void render() {
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		populateUI(); //defined in ui.h, implemented in ui.cpp

		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(currentGLFWWindow, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		//Show the newly rendered content and replace the buffer
		glfwSwapBuffers(currentGLFWWindow);
	}


	void begin()
	{
		setup();

		while (!glfwWindowShouldClose(currentGLFWWindow))
		{
			render();
			glfwPollEvents();
		}

		destroy();
	}

	float estimateSystemScale()
	{
		//Set scale based on scale of monitor
		GLFWmonitor* monitor = glfwGetPrimaryMonitor();
		float scale = 2.f;
		glfwGetMonitorContentScale(monitor, &scale, nullptr);
		return scale;
	}

	GLFWwindow* getCurrentWindow()
	{
		return nullptr;
	}

	void setSwapInterval(int newSwapInterval)
	{
		glfwSwapInterval(newSwapInterval);
	}

	void setClearColor(int red, int green, int blue, int alpha)
	{
		clear_color.w = red;
		clear_color.x = green;
		clear_color.y = blue;
		clear_color.z = alpha;
	}

	void setGuiScale(float guiScale) {
		int fbw, fbh, ww, wh;
		glfwGetFramebufferSize(currentGLFWWindow, &fbw, &fbh);
		glfwGetWindowSize(currentGLFWWindow, &ww, &wh);
		float pixelRatio = fbw / ww;
		ImGui::GetIO().FontGlobalScale = guiScale / pixelRatio;
	}



}