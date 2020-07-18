#include "ui.h"

namespace gui {

	float windowScale = 1.f;
	int swapInterval = 1;

	Renderer renderer;

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
			showRenderSettingsEditor = true,
			showImageViewer = false;
		//Placeholders (to be implemented later)
		SimpleCPURaytracer scpur;
		int renderWidth, renderHeight;
		float imageScale = 1.f;

		void createMenuBar() {
			if (ImGui::BeginMainMenuBar())
			{
				if (ImGui::BeginMenu("Windows")) {
					ImGui::MenuItem("Image Viewer", 0, &showImageViewer);
					ImGui::MenuItem("Render Settings", 0, &showRenderSettingsEditor);
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

		void createRenderSettingsWindow() {
			// ImGui::SetNextWindowSize(ImVec2(320,240));
			ImGui::Begin("Render Settings", &showRenderSettingsEditor);

			//Render settings
			ImGui::Separator();
			ImGui::Text("Render configuration");
			//Render size
			ImGui::Text("Render size (pixels)");
			if (scpur.busy()) {
				char str[128];
				sprintf_s(str, "%d x %d", renderWidth, renderHeight);
				ImGui::Text(str);
			}
			else {
				ImGui::DragInt("Render width", &renderWidth, 1.f, 32, 16386);
				ImGui::DragInt("Render height", &renderHeight, 1.f, 32, 16386);
				if (renderWidth < 32)renderWidth = 32;
				if (renderHeight < 32)renderHeight = 32;
			}

			ImGui::Separator();

			if (scpur.busy()) {
				if (ImGui::Button("Cancel render")) scpur.cancelRender();
				ImGui::Text("Pretending to render an image...");
				ImGui::ProgressBar(scpur.progress());
			}
			else {
				if (ImGui::Button("Start render")){
					showImageViewer = true;
					renderer.loadTexture(renderWidth, renderHeight);
					scpur.beginRendering(&renderer);
				}
				if (scpur.done()) {
					ImGui::Text("Render complete");
					char str[128];
					sprintf_s(str, "Time elapsed: %.3f seconds", scpur.getRenderTime());
					ImGui::Text(str);
				}
			}
			ImGui::End();
		}

		void creeateImageViewerWindow(){
			// Place the texture in an ImGui image
			ImGui::Begin("Image", &showImageViewer);

			if (!renderer.isInitialized()) {
				ImGui::Text("Renderer is not initialized!");
				ImGui::End();
				return;
			}


			//Display scale
			ImGui::SliderFloat("Display scale", &imageScale, 0.1, 5, "%.3f", 2.f);

			//Refresh texture
			renderer.reloadTexture();
			GLuint myTexture = renderer.getTextureID();

			//Get data about the image and texture
			ImGuiIO& io = ImGui::GetIO();
			int myW, myH;
			renderer.getDimensions(&myW, &myH);

			//Print data about the image
			ImGui::Text("Texture ID = %d", myTexture);
			ImGui::Text("size = %d x %d", myW, myH);

			//Scale the image based on the user's preference
			myW *= imageScale;
			myH *= imageScale;

			ImVec2 pos = ImGui::GetCursorScreenPos();

			//Show the image
			ImGui::Image((void*)(intptr_t)myTexture, ImVec2(myW, myH));

			//Show a magnified view of the image if hovered
			if (ImGui::IsItemHovered())
			{

				//Create variables that will be used later
				ImVec2 uv_min = ImVec2(0.0f, 0.0f);                 // Top-left
				ImVec2 uv_max = ImVec2(1.0f, 1.0f);                 // Lower-right
				ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);   // No tint
				ImVec4 border_col = ImVec4(1.0f, 1.0f, 1.0f, 0.5f); // 50% opaque white

				ImGui::BeginTooltip();
				float region_sz = 64.0f;
				float region_x = io.MousePos.x - pos.x - region_sz * 0.5f;
				float region_y = io.MousePos.y - pos.y - region_sz * 0.5f;
				float zoom = 4.0f;
				if (region_x < 0.0f) { region_x = 0.0f; }
				else if (region_x > myW - region_sz) { region_x = myW - region_sz; }
				if (region_y < 0.0f) { region_y = 0.0f; }
				else if (region_y > myH - region_sz) { region_y = myH - region_sz; }
				ImGui::Text("Min: (%.2f, %.2f)", region_x, region_y);
				ImGui::Text("Max: (%.2f, %.2f)", region_x + region_sz, region_y + region_sz);
				ImVec2 uv0 = ImVec2((region_x) / myW, (region_y) / myH);
				ImVec2 uv1 = ImVec2((region_x + region_sz) / myW, (region_y + region_sz) / myH);
				ImGui::Image((void*)(intptr_t)myTexture, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, tint_col, border_col);
				ImGui::EndTooltip();
			}

			ImGui::End();
		}
	}


	void populateUI()
	{
		UI::createMenuBar();
		if (UI::showDemoWindow) ImGui::ShowDemoWindow();
		if (UI::showImageViewer) UI::creeateImageViewerWindow();
		if (UI::showRenderSettingsEditor) UI::createRenderSettingsWindow();
		if (UI::showStyleEditor) ImGui::ShowStyleEditor();
		if (UI::showWindowSettingsEditor) UI::createWindowSettingsWindow();
	}

	void initUI()
	{
		windowScale = estimateSystemScale();
		setGuiScale(windowScale);

		renderer.loadTexture(640, 480);
		renderer.getDimensions(&UI::renderWidth, &UI::renderHeight);
	}
	
}