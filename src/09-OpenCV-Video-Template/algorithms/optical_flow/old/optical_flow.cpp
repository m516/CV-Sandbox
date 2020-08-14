#include <stdio.h>
#include <assert.h>
#include <iostream>


#include "optical_flow.hpp"


void computePixels(const unsigned char* const frame1, const unsigned char* const frame2, const int width, const int height, uint16_t* outputFlow){
	int index = 0;//blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 1;//blockDim.x * gridDim.x;

	int numElements = width*height;

	index*=2;
	stride*=2;
	numElements*=2;
	
	for(int i = index; i < numElements; i += stride){
		//TODO
		outputFlow[i]=-128;
		outputFlow[i+1]=255;
	}
}


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
	//Get the dimensions of the image
	int width = flow.cols, height = flow.rows;
	//Copy the data from flow to a new array called flow_device. This will make it easier to change to CUDA
	uint16_t* flow_device = (uint16_t*)malloc(width*height*2*sizeof(uint16_t));
	//Upload the data
	memcpy(flow_device, flow.data, width*height*2*sizeof(uint16_t));
	//Compute the pixels
	computePixels(frame1.data, frame2.data, width, height, flow_device);
	//Download the result
	memcpy(flow.data, flow_device, width*height*2*sizeof(uint16_t));
	free(flow_device);

	return true;
}

bool OpticalFlow::invalid()
{
	//Don't do anything if the image dimensions don't match.
	return frame1.rows != frame2.rows || frame1.cols != frame2.cols;
}
