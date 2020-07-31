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
	flow = Mat(100, 100, CV_16SC2);
	randu(flow, Scalar(0, 0, 0), Scalar(255, 255, 255));

	return true;
}

bool OpticalFlowCPU::calculateOpticalFlowWithNewFrame(Mat& newFrame){
	return calculateOpticalFlow(newFrame, frame1);
}

void OpticalFlowCPU::addToGUI()
{
	if (viewersinitialized) {
		frame1Viewer.addToGUI();
		frame2Viewer.addToGUI();
		flowViewer.addToGUI();
	}
	else {
		ImGui::Text("OpticalFlow not initialized");
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
	else flowViewer = MatViewer("Flow", flow);

	viewersinitialized = true;
}

