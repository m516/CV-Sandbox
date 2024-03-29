#pragma once
#include "glad/glad.h"
#include "GLFW/glfw3.h"

namespace App {
	class Window {
	public:
		Window();
		Window(int width, int height);
		bool open() { return _open; }
		bool shouldClose();
		void refresh();
		void close();
		void clear();

		GLFWwindow* window() { return _window; }
		operator GLFWwindow*() { return _window; }
		void resize(int width, int height, bool doResize = true);
		void getDimensions(int& w, int& h);
		int getWidth();
		int getHeight();
		float getAspectRatio();


	private:
		GLFWwindow* _window;
		bool _open = false;
		float _framerate = 0.0;
	};
}