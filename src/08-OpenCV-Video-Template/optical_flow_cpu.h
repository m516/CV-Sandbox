#pragma once
#include <opencv2/opencv.hpp>
#include "mat_viewer.h"
using namespace cv;

class OpticalFlowCPU {
public:
	OpticalFlowCPU();
	void addToGUI();
	bool calculateOpticalFlow(Mat& mat1, Mat& mat2);
	bool calculateOpticalFlowWithNewFrame(Mat& newFrame);
	void initOrUpdateViewers();
private:
	Mat frame1, frame2;
	Mat flow;

	MatViewer frame1Viewer, frame2Viewer, flowViewer;
	
	bool viewersinitialized = false;
};