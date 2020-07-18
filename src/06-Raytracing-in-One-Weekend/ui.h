#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

#include "app.h"
#include "renderer.h"
#include "raytracing/simpleCPURenderer.h"

namespace gui {
	void setStyle();
	void populateUI();
	void initUI();
}