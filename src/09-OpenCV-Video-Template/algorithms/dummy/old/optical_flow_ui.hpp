#pragma once
#include <opencv2/opencv.hpp>
#include "../../mat_viewer.hpp"
#include "optical_flow.hpp"

using namespace cv;

class OpticalFlowUI {
public:
	Mat* imageSource;
	OpticalFlowUI(Mat& imageSource) {
		this->imageSource = &imageSource;
	}
	void addToGUI();
private:
	OpticalFlow opticalFlow;
	Mat visibleFlow;


	void initOrUpdateViewers();
	void initOrUpdateFlow();

	MatViewer frame1Viewer, frame2Viewer, flowViewer;
	
	bool viewersinitialized = false;

	int previousState = -1;
};