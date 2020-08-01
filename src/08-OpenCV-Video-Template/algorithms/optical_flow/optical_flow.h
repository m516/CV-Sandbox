#pragma once
#include <opencv2/opencv.hpp>
#include "optical_flow.h"

using namespace cv;

class OpticalFlow {
public:
	OpticalFlow();


	bool calculateOpticalFlow(Mat& mat1, Mat& mat2);
	bool calculateOpticalFlowWithNewFrame(Mat& newFrame);
	bool recaulculateOpticalFlow();

	Mat frame1, frame2;
	Mat flow;
private:
	bool invalid();
};