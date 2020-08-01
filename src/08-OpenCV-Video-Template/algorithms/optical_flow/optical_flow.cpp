#include "optical_flow.h"

OpticalFlow::OpticalFlow() {}

bool OpticalFlow::calculateOpticalFlow(Mat& mat1, Mat& mat2) {

	//Update the fields in this optical flow 
	mat2.copyTo(frame2);
	mat1.copyTo(frame1);

	if (invalid()) return false;

	//Create a random matrix for the flow
	flow = Mat(mat1.rows, mat1.cols, CV_16SC2);

	recaulculateOpticalFlow();

	return true;
}

bool OpticalFlow::calculateOpticalFlowWithNewFrame(Mat& newFrame) {
	return calculateOpticalFlow(newFrame, frame1);
}

bool OpticalFlow::recaulculateOpticalFlow() {
	//Can't recalculate the optical flow if the two frames are incompatible
	if (invalid()) return false;
	//Initial calculation hasn't been done yet if the flow matrix's dimensions are 0
	if (flow.rows == 0 || flow.cols == 0) return false;

	//Generate a new vector field. This is a placeholder for other calculations that come later.
	randu(flow, Scalar(-255, -255), Scalar(255, 255));

	for (int i = 0; i < flow.rows; i++) {
		for (int j = 0; j < flow.cols; j++) {

		}
	}

	return true;
}

bool OpticalFlow::invalid()
{
	//Don't do anything if the image dimensions don't match.
	return frame1.rows != frame2.rows || frame1.cols != frame2.cols;
}
