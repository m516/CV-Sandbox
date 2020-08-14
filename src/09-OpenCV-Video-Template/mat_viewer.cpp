#include "mat_viewer.hpp"

#include <imgui/imgui.h>


// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint for use
GLuint MatViewer::matToTexture(const cv::Mat& mat) {
	// Generate a number for our textureID's unique handle
	GLuint textureID;
	glGenTextures(1, &textureID);

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = GL_BGR;
	if (mat.channels() == 1)
	{
		inputColourFormat = GL_LUMINANCE;
	}

	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
		0,                 // Pyramid level (for mip-mapping) - 0 is the top level
		GL_RGB,            // Internal colour format to convert to
		mat.cols,          // Image width  i.e. 640 for Kinect in standard mode
		mat.rows,          // Image height i.e. 480 for Kinect in standard mode
		0,                 // Border width in pixels (can either be 1 or 0)
		inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
		GL_UNSIGNED_BYTE,  // Image data type
		mat.ptr());        // The actual image data itself

	return textureID;
}


void MatViewer::reloadTexture()
{
	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexSubImage2D(
		GL_TEXTURE_2D,
		0,
		0,
		0,
		mat->cols,
		mat->rows,
		GL_BGR,
		GL_UNSIGNED_BYTE,
		mat->ptr()
	);

	glBindTexture(GL_TEXTURE_2D, 0);
}


MatViewer::MatViewer(std::string name, Mat& mat)
{

	cout << "Mat: " << mat.cols << " x " << mat.rows << endl;

	this->name = name;
	this->mat = &mat;

	width = mat.cols;
	height = mat.rows;

	generateTexture();
}

MatViewer::~MatViewer()
{
	//if (textureID) glDeleteTextures(1, &textureID);
}

void MatViewer::addToGUI(bool withControls, bool withTooltip)
{
	//Skip if not initialized
	if(!initialized()) {
		ImGui::Text("MatViewer not initialized");
		return;
	}

	ImGui::Text(name.c_str());

	//Display scale
	ImGui::SliderFloat((name + " Display Scale").c_str(), &imageScale, 0.1, 5, "%.3f", 2.f);

	//Get data about the image and texture
	ImGuiIO& io = ImGui::GetIO();
	float myW = (float)width,
	myH = (float)height;

	//Scale the image based on the user's preference
	myW *= imageScale;
	myH *= imageScale;

	ImVec2 pos = ImGui::GetCursorScreenPos();

	//Show the image
	ImGui::Image((void*)(intptr_t)textureID, ImVec2(myW, myH));

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
		ImGui::Image((void*)(intptr_t)textureID, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, tint_col, border_col);
		ImGui::EndTooltip();
	}

	//Print data about the image
	ImGui::Text("Texture ID = %d", textureID);
	ImGui::Text("size = %d x %d", width, height);

}

void MatViewer::update()
{
	if (mat == nullptr) return;
	if(mat->rows != height || mat->cols != width) generateTexture();
	else reloadTexture();
}

void MatViewer::generateTexture()
{

	if (mat->empty()) {
		std::cout << "image empty" << std::endl;
		return;
	}

	//Destroy the last texture
	if(textureID) glDeleteTextures(1, &textureID);


	//Generate the texture
	textureID = matToTexture(*mat);

	//Update the dimensions
	width = mat->cols;
	height = mat->rows;

	cout << "TextureID = " << textureID << endl;

}
