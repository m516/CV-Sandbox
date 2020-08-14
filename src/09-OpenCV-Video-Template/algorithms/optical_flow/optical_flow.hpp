#pragma once
#include <opencv2/opencv.hpp>
#include "optical_flow.hpp"

using namespace cv;

class OpticalFlow {
public:
	OpticalFlow();


	bool calculateOpticalFlow(Mat& mat1, Mat& mat2);
	bool calculateOpticalFlowWithNewFrame(Mat& newFrame);
	bool recaulculateOpticalFlow();
	/*frame1 and frame2 are both BGR images whose colors are 8-bits each. They must have the same dimensions for OpticalFlow calculations to work correctly*/
	Mat frame1, frame2;
	/*A matrix of 2D vectors*/
	Mat flow;
private:
	bool invalid();
};