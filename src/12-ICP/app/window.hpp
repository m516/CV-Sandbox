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
	private:
		GLFWwindow* _window;
		bool _open = false;
	};
}