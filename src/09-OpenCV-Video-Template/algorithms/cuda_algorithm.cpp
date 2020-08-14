#include "cuda_algorithm.hpp"
#include <imgui/imgui.h>


void CUDAVisionAlgorithm::setInput(Mat& input)
{
    if (input.empty()) return;

    d_imageInputData = GpuMat(input);
    alreadyProcessed = false;
}

void CUDAVisionAlgorithm::getOutput(Mat& output)
{
    d_imageOutputData.download(output);
}

void CUDAVisionAlgorithm::addToGUI(){
    if (ImGui::Button("Process me!")) process();
    ImGui::Separator();
    if (d_imageInputData.cols > 0 || d_imageInputData.rows > 0) {
        imageInputTexture.copyFrom(d_imageInputData);
        ImGui::Image((ImTextureID)imageInputTexture.texId(), ImVec2(d_imageInputData.rows, d_imageInputData.cols));
    }
    else {
        ImGui::Text("No input data to show here.");
    }
    if (d_imageOutputData.cols > 0 || d_imageOutputData.rows > 0) {
        imageOutputTexture.copyFrom(d_imageOutputData);
        ImGui::Image((ImTextureID)imageOutputTexture.texId(), ImVec2(d_imageOutputData.rows, d_imageOutputData.cols));
    }
    else {
        ImGui::Text("No output data to show here.");
    }
}