#include "GUI.hpp"


#include <thread>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"


#include "window.hpp"
#include "property.hpp"




PointCloud::Data<6> pointCloud;
PointCloud::Renderer pointCloudRenderer;
App::Window window;


bool showViewControls = true;
bool showDemoWindow = false;

void setStyle() {
	ImGuiStyle* style = &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 0.70f);
	colors[ImGuiCol_ChildBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.21f, 0.22f, 0.54f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.40f, 0.40f, 0.40f, 0.40f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.18f, 0.18f, 0.18f, 0.67f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.29f, 0.29f, 0.29f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(1.00f, 0.67f, 0.00f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.52f, 0.35f, 0.00f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(1.00f, 0.65f, 0.00f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.44f, 0.44f, 0.44f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.54f, 0.55f, 0.57f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.50f, 0.38f, 0.00f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.70f, 0.70f, 0.70f, 0.31f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.70f, 0.70f, 0.70f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.48f, 0.50f, 0.52f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.72f, 0.72f, 0.72f, 0.78f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.91f, 0.91f, 0.91f, 0.25f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.81f, 0.81f, 0.81f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.46f, 0.46f, 0.46f, 0.95f);
	colors[ImGuiCol_Tab] = ImVec4(0.31f, 0.32f, 0.33f, 0.86f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.49f, 0.52f, 0.54f, 0.80f);
	colors[ImGuiCol_TabActive] = ImVec4(0.50f, 0.40f, 0.05f, 1.00f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.07f, 0.10f, 0.15f, 0.97f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.14f, 0.26f, 0.42f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.42f, 0.31f, 0.13f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.58f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.40f, 0.25f, 0.05f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.87f, 0.87f, 0.87f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 0.56f, 0.00f, 1.00f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);



	style->FramePadding = ImVec2(4, 2);
	style->ItemSpacing = ImVec2(16, 16);
	style->IndentSpacing = 12;
	style->ScrollbarSize = 20;

	style->WindowRounding = 4;
	style->FrameRounding = 4;
	style->ScrollbarRounding = 6;
	style->GrabRounding = 4;

	style->WindowTitleAlign = ImVec2(1.0f, 0.5f);

	style->DisplaySafeAreaPadding = ImVec2(4, 4);

}



/**
 * A helper function for terminating the program
 */
void terminate(int errorCode) {
	using namespace std;
	cout << "Closing application";
	//Exit
	exit(errorCode);
}



void viewControls()
{

	ImGui::Begin("Window Settings", &showViewControls);

	static Soft::Property<float> cameraPosition[] = { -2, 1, 2 };
	static Soft::Property<float>  cameraCenter[] = { 0, .5, 0 };

	ImGui::SliderFloat("Camera position x", cameraPosition[0], -10, 10);
	ImGui::SliderFloat("Camera position y", cameraPosition[1], -10, 10);
	ImGui::SliderFloat("Camera position z", cameraPosition[2], -10, 10);
	ImGui::SliderFloat("Camera center x",   cameraCenter[0],   -10, 10);
	ImGui::SliderFloat("Camera center y",   cameraCenter[1],   -10, 10);
	ImGui::SliderFloat("Camera center z",   cameraCenter[2],   -10, 10);

	for (int i = 0; i < 3; i++) {
		cameraPosition[i].stepExp(0.2);
		cameraCenter[i].stepExp(0.2);
	}

	float cp[] = { cameraPosition[0], cameraPosition[1], cameraPosition[2] };
	float cc[] = { cameraCenter[0], cameraCenter[1], cameraCenter[2] };

	pointCloudRenderer.setViewLookAt(cp, cc);

	ImGui::End();
}


void mainMenu()
{

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Windows")) {
			ImGui::MenuItem("View Controls", 0, &showViewControls);
			ImGui::MenuItem("Demo Window", 0, &showDemoWindow);
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}

	viewControls();
	ImGui::ShowDemoWindow(&showDemoWindow);
}





namespace App {

	namespace UI {
		void init() {
			window = Window(1080, 1080);
			if (!window.open()) terminate(1);
		}
		void UI::setPointCloud(std::string filename)
		{
			pointCloud = PointCloud::loadXYZRGB(filename);
			pointCloudRenderer.setSparse(pointCloud, 0.1);
			//pointCloudRenderer.set(pointCloud);
		}
		void UI::run()
		{
			// Setup Dear ImGui context
			IMGUI_CHECKVERSION();
			ImGui::CreateContext();
			ImGuiIO& io = ImGui::GetIO(); (void)io;
			ImGui_ImplGlfw_InitForOpenGL(window, true);
			ImGui_ImplOpenGL3_Init("#version 130");
			setStyle();

			while (window.open() && !window.shouldClose()) {

				ImGui_ImplOpenGL3_NewFrame();
				ImGui_ImplGlfw_NewFrame();
				ImGui::NewFrame();


				mainMenu();

				window.clear();
				pointCloudRenderer.display();
				ImGui::Render();
				ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
				window.refresh();
			}
			window.close();
		}
	}



	
	
}