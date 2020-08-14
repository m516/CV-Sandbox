#include "cuda_algorithm.hpp"
#include <imgui/imgui.h>


void CUDAVisionAlgorithm::setInput(Mat& input)
{
    d_imageInputData.upload(input);
}

void CUDAVisionAlgorithm::getOutput(Mat& output)
{
    d_imageOutputData.download(output);
}

void CUDAVisionAlgorithm::addToGUI(){
    if (d_imageInputData.cols > 0 || d_imageInputData.rows > 0) {
        imageInputTexture.copyFrom(d_imageInputData);
        ImGui::Image((ImTextureID)imageInputTexture.texId(), ImVec2(d_imageInputData.rows, d_imageInputData.cols));
    }
    if (d_imageOutputData.cols > 0 || d_imageOutputData.rows > 0) {
        imageOutputTexture.copyFrom(d_imageOutputData);
        ImGui::Image((ImTextureID)imageOutputTexture.texId(), ImVec2(d_imageOutputData.rows, d_imageOutputData.cols));
    }
}