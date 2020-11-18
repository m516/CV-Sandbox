#include "optical_flow_ui.hpp"
#include "imgui/imgui.h"

#define STATUS_CALCULATE_SUCCESSFUL 0
#define STATUS_CALCULATE_NEWFRAME_SUCCESSFUL 1
#define STATUS_RECALCULATE_SUCCESSFUL 2
#define STATUS_CALCULATE_UNSUCCESSFUL 3
#define STATUS_CALCULATE_NEWFRAME_UNSUCCESSFUL 4
#define STATUS_RECALCULATE_UNSUCCESSFUL 5


/**
 * @brief Creates a color with a given hue and value (whose values range form 0-255), 
 * assuming saturation = 255
 * 
 * @param hue 
 * @param value 
 * @return Color 
 */
void hv2rgb(uint8_t hue, uint8_t value, uint8_t* outRed, uint8_t* outGreen, uint8_t* outBlue)
{
	if(hue<0) hue = 0;
	if(hue>255) hue = 255;
	if(value<0) value=0;
	if(value>255) value=255;

	int i = hue/43;
	float j = (float)(hue%43);
	j/=43;
	float k = 1-j;
	float r, g, b;
	switch(i){
		case 0:
			r=1;
			g=j;
			b=0;
			break;
		case 1:
			r=k;
			g=1;
			b=0;
			break;
		case 2:
			r=0;
			g=1;
			b=j;
			break;
		case 3:
			r=0;
			g=k;
			b=1;
			break;
		case 4:
			r=j;
			g=0;
			b=1;
			break;
		case 5:
			r=1;
			g=0;
			b=k;
			break;
	}
	*outRed = (uint8_t)(r*value);
	*outGreen = (uint8_t)(g*value);
	*outBlue = (uint8_t)(b*value);
	return;
}

void positionToColor(float x, float y, uint8_t* outRed, uint8_t* outGreen, uint8_t* outBlue){
	float hue = atan2f(x, y) * 40.584510488433310f;
	float value = min(sqrtf(x*x+y*y)* 0.70434464532253757313365f, 255.f);

	return hv2rgb((uint8_t)hue, (uint8_t)value, outRed, outGreen, outBlue);
}



void OpticalFlowUI::addToGUI()
{
	//Calculate flow button
	if (ImGui::Button("Calculate Flow")) {
		if (opticalFlow.calculateOpticalFlowWithNewFrame(*imageSource)) previousState = STATUS_CALCULATE_NEWFRAME_SUCCESSFUL;
		else previousState = STATUS_CALCULATE_NEWFRAME_UNSUCCESSFUL;
		initOrUpdateViewers();
	}
	//Recalculate flow button
	if (ImGui::Button("Recalculate")) {
		if (opticalFlow.recaulculateOpticalFlow()) previousState = STATUS_RECALCULATE_SUCCESSFUL;
		else previousState = STATUS_RECALCULATE_UNSUCCESSFUL;
		initOrUpdateFlow();
	}
	//Status text
	switch (previousState) {
	case STATUS_CALCULATE_SUCCESSFUL:
		ImGui::Text("Operation status: STATUS_CALCULATE_SUCCESSFUL");
		break;
	case STATUS_CALCULATE_NEWFRAME_SUCCESSFUL:
		ImGui::Text("Operation status: STATUS_CALCULATE_NEWFRAME_SUCCESSFUL");
		break;
	case STATUS_RECALCULATE_SUCCESSFUL:
		ImGui::Text("Operation status: STATUS_RECALCULATE_SUCCESSFUL");
		break;
	case STATUS_CALCULATE_UNSUCCESSFUL:
		ImGui::Text("Operation status: STATUS_CALCULATE_UNSUCCESSFUL");
		break;
	case STATUS_CALCULATE_NEWFRAME_UNSUCCESSFUL:
		ImGui::Text("Operation status: STATUS_CALCULATE_NEWFRAME_UNSUCCESSFUL");
		break;
	case STATUS_RECALCULATE_UNSUCCESSFUL:
		ImGui::Text("Operation status: STATUS_RECALCULATE_UNSUCCESSFUL");
		break;
	default:
		ImGui::Text("Operation status: Nothing to report yet");
	}
	//Don't render anything else if not initialized properly
	if(!viewersinitialized){
		ImGui::Text("OpticalFlow not initialized");
		return;
	}
	

	/*Image viewers*/
	if (ImGui::TreeNode("Frame 1"))
	{
		frame1Viewer.addToGUI();
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Frame 2"))
	{
		frame2Viewer.addToGUI();
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Calculated flow"))
	{
		flowViewer.addToGUI();
		ImGui::TreePop();
	}
	
}

void OpticalFlowUI::initOrUpdateViewers()
{

	if(visibleFlow.rows != opticalFlow.flow.rows || visibleFlow.cols != opticalFlow.flow.cols)
		visibleFlow = cv::Mat(opticalFlow.flow.rows, opticalFlow.flow.cols, CV_8UC3);

	if (frame1Viewer.initialized())
		frame1Viewer.update();
	else frame1Viewer = MatViewer("Frame 1", opticalFlow.frame1);

	if (frame2Viewer.initialized())
		frame2Viewer.update();
	else frame2Viewer = MatViewer("Frame 2", opticalFlow.frame2);

	initOrUpdateFlow();

	viewersinitialized = true;
}

void OpticalFlowUI::initOrUpdateFlow()
{
	//Don't update flow if other viewers aren't initialized
	if (!viewersinitialized) return;

	if (!flowViewer.initialized()) flowViewer = MatViewer("Flow", visibleFlow);

	//Update the visible optical flow
	// We iterate over all pixels of the image
	for (int r = 0; r < visibleFlow.rows; r++) {
		// We obtain a pointer to the beginning of row r for flow and visualFlow
		cv::Vec2s* ptr = opticalFlow.flow.ptr<cv::Vec2s>(r);
		cv::Vec3b* vptr = visibleFlow.ptr<cv::Vec3b>(r);

		for (int c = 0; c < opticalFlow.flow.cols; c++) {
			positionToColor(ptr[c][0], ptr[c][1], &vptr[c][2], &vptr[c][1], &vptr[c][0]);
		}
	}

	flowViewer.update();
}

