#pragma once

#include <glad.h>
#include <cuda_gl_interop.h>
#include <opencv2/core.hpp>
#include <stdio.h>

using namespace cv;
using namespace cv::cuda;


/**
 * @brief An abstract class that generalizes most CV algorithms that can be implemented with CUDA.
 * 
 * 
 */
class CUDAVisionAlgorithm {
public:
    void setInput(const Mat& input);
    void getOutput(Mat& output);
    virtual bool process(){dirty = true; alreadyProcessed=true; return true;};
    virtual void addToGUI();
    bool empty(){return imageInputTexture==0;}
    bool alreadyProcessed = false;
protected:
    /**
     * @brief True if the textures and surfaces are initialized. 
     * 
     * 
     * Prevents memory leaks
     */
    bool surfacesInitialized = false;
    /**
     * @brief The width and height of a texture/surface pair.
     * 
     */
    struct ImgDim{int width, height;};
    /**
     * @brief Create a GL texture and a CUDA surface that are bound to each other, from data.
     * 
     * @param dimensions the width and height of the image to copy into the texture/surface pair.
     * @param height the height of the image to copy into the texture/surface pair.
     * @param data the data in unsigned 8-bit, BGR data.
     * @param textureOut an address to a GL texture reference
     * @param surfaceOut an address to a CUDA surface reference
     */
    void createTextureSurfacePair(const CUDAVisionAlgorithm::ImgDim& dimensions, uint8_t* const data, GLuint& textureOut, cudaGraphicsResource_t& graphicsResourceOut, cudaSurfaceObject_t& surfaceOut);
    /**
     * @brief Destroys all textures and CUDA data, freeing all the memory bound to them.
     * 
     */
    void destroyEverything();
    /**
     * @brief Set the Output Dimensions based on the input dimensions.
     * Called immediately after setInput.
     */
    virtual void setOutputDimensions() = 0;
    /**
     * @brief The dimensions of an image and its corresponding texture.
     * 
     */
    ImgDim imageInputDimensions, imageOutputDimensions;
    /**
     * @brief A CUDA surface that can be read to, written from, or synchronized with a Mat or
     * OpenGL texture
     * 
     */
    cudaSurfaceObject_t d_imageInputTexture = 0, d_imageOutputTexture = 0;
    /**
     * @brief A CUDA resource that's bound to an array in CUDA memory
     */
    cudaGraphicsResource_t d_imageInputGraphicsResource, d_imageOutputGraphicsResource;
    /**
     * @brief A renderable OpenGL texture that is synchronized with the CUDA data
     * @see d_imageInputTexture, d_imageOutputTexture
     */
    GLuint imageInputTexture = 0, imageOutputTexture = 0;
    /**
     * @brief True if this Mat has been processed.
     * 
     */
    bool dirty = false;
};