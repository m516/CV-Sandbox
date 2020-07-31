#include "ui.h"
#include "video_viewer.h"
#include "optical_flow_cpu.h"

namespace gui {

	float windowScale = 1.f;
	int swapInterval = 1;


	//Project-specific fields
	VideoCapture videoCapture;
	VideoViewer* videoViewer;
	OpticalFlowCPU opticalFlowCPU;


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

	namespace UI {

		bool showStyleEditor = false,
			showDemoWindow = false,
			showWindowSettingsEditor = false,
			showVideoViewer = true,
			showOpticalFlow = true;


		void createMenuBar() {
			if (ImGui::BeginMainMenuBar())
			{
				if (ImGui::BeginMenu("Windows")) {
					ImGui::MenuItem("Video Viewer", 0, &showVideoViewer);
					ImGui::MenuItem("Optical Flow", 0, &showOpticalFlow);
					ImGui::MenuItem("Window Settings", 0, &showWindowSettingsEditor);
					ImGui::Separator();
					ImGui::MenuItem("Style editor", 0, &showStyleEditor);
					ImGui::MenuItem("Demo Window", 0, &showDemoWindow);
					ImGui::EndMenu();
				}
				ImGui::EndMainMenuBar();
			}
		}

		void createWindowSettingsWindow() {
			// ImGui::SetNextWindowSize(ImVec2(320,240));
			ImGui::Begin("Window Settings", &showWindowSettingsEditor);
			// Gui rendering size
			if (ImGui::SliderFloat("Display scale", &windowScale, 1, 3)) setGuiScale(windowScale);
			// Image rendering size relative to GUI size
			//ImGui::SliderFloat("Image scale", &imageScale, 0.1, 4);
			// Framerate division factor
			if (ImGui::SliderInt("Swap interval", &swapInterval, 1, 5)) setSwapInterval(swapInterval);
			//Stats
			ImGui::Text("Stats:");
			// Framerate
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
		}
			
		void createImageViewerWindow(){
			// Place the texture in an ImGui image
			ImGui::Begin("Video", &showVideoViewer);

			videoViewer->addToGUI();

			ImGui::End();
		}

		void createOpticalFlowViewer(){
			ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
			ImGui::Begin("Optical Flow", &showVideoViewer, flags);
			
			if (ImGui::Button("Calculate Flow")) {
				opticalFlowCPU.calculateOpticalFlowWithNewFrame(videoViewer->mat);
				opticalFlowCPU.initOrUpdateViewers();
			}

			opticalFlowCPU.addToGUI();		

			ImGui::End();
		}
	}


	void populateUI()
	{
		UI::createMenuBar();
		if (UI::showDemoWindow) ImGui::ShowDemoWindow();
		if (UI::showVideoViewer) UI::createImageViewerWindow();
		if (UI::showStyleEditor) ImGui::ShowStyleEditor();
		if (UI::showWindowSettingsEditor) UI::createWindowSettingsWindow();
		if (UI::showOpticalFlow) UI::createOpticalFlowViewer();
	}

	void initUI()
	{
		windowScale = estimateSystemScale();
		setGuiScale(windowScale);

		//Load the video
		// Get an image name 
		std::string filename = MEDIA_DIRECTORY;
		// MEDIA_DIRECTORY is defined in the root-level CMake script
		filename += "CameraOrbit.MOV";
		videoCapture = VideoCapture(filename);

		videoViewer = new VideoViewer(videoCapture);

	}
	
}