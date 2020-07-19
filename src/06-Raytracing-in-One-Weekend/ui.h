#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "app.h"
#include "renderer.h"
#include "raytracing/simpleCPURenderer.h"
#include "raytracing/raytracers/raytracers.h"

namespace gui {
	/*Sets the look-and-feel of the UI */
	void setStyle();
	/*
	This function is responsible for the layout of the entire user interface. 
	
	A function under "app.cpp" updates the window periodically, at a refresh rate set by the user.
	Then it creates a new ImGui frame.
	Then it runs this function.
	Then it calls ImGui::Render()
	Then it renders the ImGui frame on the OpenGL pixel buffer using ImGui_ImplOpenGL3_RenderDrawData() to show the GUI to the user.

	Therefore, this function is not at all responsible for rendering the UI, yet 
	it controls everything about the struture, layout, and appearance of the UI.
	*/
	void populateUI();
	/*Initializes the UI. Must be called before "populateUI()"*/
	void initUI();
}