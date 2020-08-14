#pragma once
#include <opencv2/opencv.hpp>
#include "optical_flow.hpp"
#include "../cuda_algorithm.hpp"

using namespace cv;

class OpticalFlow : public CUDAVisionAlgorithm {
public:
	virtual bool process();
private:
	bool invalid() { return d_imageInputData.empty(); }
};