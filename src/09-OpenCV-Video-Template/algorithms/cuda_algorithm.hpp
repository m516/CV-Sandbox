#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/opengl.hpp>

using namespace cv;
using namespace cv::cuda;

#include <glad.h>

/**
 * @brief An abstract class that generalizes most CV algorithms that can be implemented with CUDA.
 * 
 * 
 */
class CUDAVisionAlgorithm {
public:
    void setInput(Mat& input);
    void getOutput(Mat& output);
    virtual void process() = 0;
    virtual void addToGUI();
private:
    GpuMat d_imageInputData;
    GpuMat d_imageOutputData; 
    cv::ogl::Texture2D imageInputTexture, imageOutputTexture;;
};