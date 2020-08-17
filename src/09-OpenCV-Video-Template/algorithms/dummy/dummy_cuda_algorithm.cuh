#pragma once
#include "../cuda_algorithm.cuh"
#include <opencv2/core.hpp>

using namespace cv;

class DummyCUDAAlgorithm : public CUDAVisionAlgorithm {
public:
	virtual bool process();
	virtual void setOutputDimensions(){
		imageOutputDimensions.width=imageInputDimensions.width; imageOutputDimensions.height=imageInputDimensions.height;
		}
private:
	bool invalid() { return imageInputTexture==0;}
};