#include "optical_flow_cpu.h"
#include "imgui/imgui.h"

OpticalFlowCPU::OpticalFlowCPU(){};

bool OpticalFlowCPU::calculateOpticalFlow(Mat& mat1, Mat& mat2){

	//Update the fields in this optical flow 
	mat2.copyTo(frame2);
	mat1.copyTo(frame1);

	//Don't do anything if the image dimensions don't match.
	if(mat1.rows != mat2.rows || mat1.cols != mat2.cols) return false;

	//Create a random matrix for the flow
	flow = Mat(mat1.rows, mat1.cols, CV_16SC2);
	visibleFlow = Mat(mat1.rows, mat1.cols, CV_8UC3);

	recaulculateOpticalFlow();

	return true;
}

bool OpticalFlowCPU::calculateOpticalFlowWithNewFrame(Mat& newFrame){
	return calculateOpticalFlow(newFrame, frame1);
}

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

void OpticalFlowCPU::recaulculateOpticalFlow(){

	randu(flow, Scalar(-255, -255, -255), Scalar(255, 255, 255));

	for(int i = 0; i < flow.rows; i++){
		for(int j = 0; j < flow.cols; j++){
			
		}
	}

	//Update the visible optical flow
	// We iterate over all pixels of the image
    for(int r = 0; r < flow.rows; r++) {
        // We obtain a pointer to the beginning of row r for flow and visualFlow
        cv::Vec2s* ptr = flow.ptr<cv::Vec2s>(r);
        cv::Vec3b* vptr = visibleFlow.ptr<cv::Vec3b>(r);

        for(int c = 0; c < flow.cols; c++) {
			positionToColor(ptr[c][0], ptr[c][1], &vptr[c][2], &vptr[c][1], &vptr[c][0]);
        }
    }
}

void OpticalFlowCPU::addToGUI()
{
	
	if(!viewersinitialized){
		ImGui::Text("OpticalFlow not initialized");
		return;
	}
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
	if(ImGui::Button("Recalculate")) {
		recaulculateOpticalFlow();
		initOrUpdateViewers();
	}
}

void OpticalFlowCPU::initOrUpdateViewers()
{
	if (frame1Viewer.initialized())
		frame1Viewer.update();
	else frame1Viewer = MatViewer("Frame 1", frame1);

	if (frame2Viewer.initialized())
		frame2Viewer.update();
	else frame2Viewer = MatViewer("Frame 2", frame2);

	if (flowViewer.initialized())
		flowViewer.update();
	else flowViewer = MatViewer("Flow", visibleFlow);

	viewersinitialized = true;
}

