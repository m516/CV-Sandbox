#include "cuda_algorithm.cuh"
#include <imgui/imgui.h>
#include <iostream>
#include <string>



#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
      printf("Cuda error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }


void CUDAVisionAlgorithm::createTextureSurfacePair(const CUDAVisionAlgorithm::ImgDim& dimensions, uint8_t* const data, GLuint& textureOut, cudaGraphicsResource_t& graphicsResourceOut, cudaSurfaceObject_t& surfaceOut){

    // Create the OpenGL texture that will be displayed with GLAD and GLFW
    glGenTextures(1, &textureOut);   
    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureOut);
    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // Create the texture and its attributes
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
        0,                // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGBA,          // Internal color format to convert to
        dimensions.width,            // Image width  i.e. 640 for Kinect in standard mode
        dimensions.height,           // Image height i.e. 480 for Kinect in standard mode
        0,                // Border width in pixels (can either be 1 or 0)
        GL_BGR,          // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE, // Image data type.
        data);            // The actual image data itself
    //Note that the type of this texture is an RGBA UNSIGNED_BYTE type. When CUDA surfaces
    //are synchronized with OpenGL textures, the surfaces will be of the same type.
    //They won't know or care about their data types though, for they are all just byte arrays
    //at heart. So be careful to ensure that any CUDA kernel that handles a CUDA surface
    //uses it as an appropriate type. You will see that the update_surface kernel (defined 
    //above) treats each pixel as four unsigned bytes along the X-axis: one for red, green, blue,
    //and alpha respectively.

    //Create the CUDA array and texture reference
    cudaArray *bitmap_d;
    //Register the GL texture with the CUDA graphics library. A new cudaGraphicsResource is created, and its address is placed in cudaTextureID.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g80d12187ae7590807c7676697d9fe03d
    cudaGraphicsGLRegisterImage(&graphicsResourceOut, textureOut, GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlagsNone);
    cudaCheckError();
    //Map graphics resources for access by CUDA.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gad8fbe74d02adefb8e7efb4971ee6322
    cudaGraphicsMapResources(1, &graphicsResourceOut, 0);
    cudaCheckError();
    //Get the location of the array of pixels that was mapped by the previous function and place that address in bitmap_d
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
    cudaGraphicsSubResourceGetMappedArray(&bitmap_d, graphicsResourceOut, 0, 0);
    cudaCheckError();
    //Create a CUDA resource descriptor. This is used to get and set attributes of CUDA resources.
    //This one will tell CUDA how we want the bitmap_surface to be configured.
    //Documentation for the struct: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaResourceDesc.html#structcudaResourceDesc
    struct cudaResourceDesc resDesc;
    //Clear it with 0s so that some flags aren't arbitrarily left at 1s
    memset(&resDesc, 0, sizeof(resDesc));
    //Set the resource type to be an array for convenient processing in the CUDA kernel.
    //List of resTypes: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g067b774c0e639817a00a972c8e2c203c
    resDesc.resType = cudaResourceTypeArray;
    //Bind the new descriptor with the bitmap created earlier.
    resDesc.res.array.array = bitmap_d;
    //Create a new CUDA surface ID reference.
    //This is really just an unsigned long long.
    //Docuentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gbe57cf2ccbe7f9d696f18808dd634c0a
    surfaceOut = 0;
    //Create the surface with the given description. That surface ID is placed in bitmap_surface.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT_1g958899474ab2c5f40d233b524d6c5a01
    cudaCreateSurfaceObject(&surfaceOut, &resDesc);
    cudaCheckError();
}

void CUDAVisionAlgorithm::setInput(const Mat& input)
{

    //Empty Mats cause problems in textures and surfaces.
    if (input.empty()) return;


    //Same-size images don't need texture regeneration, so skip that.
    if(input.rows == imageInputDimensions.height && input.cols == imageInputDimensions.width){

        //Use the input surface's CUDAResourceDesc to gain access to the surface data array
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        cudaGetSurfaceObjectResourceDesc(&resDesc, d_imageInputTexture);
        cudaCheckError();

        //Copy the data from the input array to the surface
        cudaMemcpyToArray(resDesc.res.array.array, 0, 0, input.data, imageInputDimensions.width* imageInputDimensions.height*3, cudaMemcpyHostToDevice);
        cudaCheckError();

        //Set status flags
        surfacesInitialized = true;
        alreadyProcessed = false;

        return;
    }


    //Clear everything that originally existed in the texture/surface
    destroyEverything();

    //Expect a BGR matrix
    if(input.type()!=CV_8UC3){
        std::string r;

        uchar depth = input.type() & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (input.type() >> CV_CN_SHIFT);

        switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
        }

        r += "C";
        r += (chans + '0');

        std::cerr << "Unexpected Mat Type: " << r << std::endl;
        throw std::invalid_argument("Invalid OpenCV matrix type. Expecting a BGR matrix (3-channel)");
    }

    //Get the size of the Mat and place it here.
    imageInputDimensions.width=input.cols;
    imageInputDimensions.height=input.rows;

    //Set the dimensions of the output matrix.
    setOutputDimensions();

    //Create the input surface/texture pair
    createTextureSurfacePair(imageInputDimensions, input.data, imageInputTexture, d_imageInputGraphicsResource, d_imageInputTexture);

    //Create the output surface/texture pair
    uint8_t* outData = new uint8_t[imageOutputDimensions.width * imageOutputDimensions.height * 3];
    createTextureSurfacePair(imageOutputDimensions, outData, imageOutputTexture, d_imageOutputGraphicsResource, d_imageOutputTexture);
    delete outData;

    //Set status flags
    surfacesInitialized = true;
    alreadyProcessed = false;
}

void CUDAVisionAlgorithm::destroyEverything(){
    if (surfacesInitialized) {

        //Input image CUDA surface
        cudaDestroySurfaceObject(d_imageInputTexture);
        cudaGraphicsUnmapResources(1, &d_imageInputGraphicsResource);
        cudaGraphicsUnregisterResource(d_imageInputGraphicsResource);
        d_imageInputTexture = 0;

        //Output image CUDA surface
        cudaDestroySurfaceObject(d_imageOutputTexture);
        cudaGraphicsUnmapResources(1, &d_imageOutputGraphicsResource);
        cudaGraphicsUnregisterResource(d_imageOutputGraphicsResource);
        d_imageOutputTexture = 0;

        //Input image GL texture
        glDeleteTextures(1, &imageInputTexture);
        imageInputTexture = 0;

        //Output image GL texture
        glDeleteTextures(1, &imageOutputTexture);
        imageOutputTexture = 0;

        surfacesInitialized = false;
    }
}

void CUDAVisionAlgorithm::getOutput(Mat& output)
{
    //TODO
    //output = cv::Mat(2, 4, CV_32F, dummy_query_data);
}

void CUDAVisionAlgorithm::addToGUI(){

    //Add a process button
    if (ImGui::Button("Process me!")) process();
    ImGui::Separator();

    //Synchronize the CUDA surfaces with the OpenGL textures, so the textures have the same data their surfaces.
    if(dirty){
        cudaDeviceSynchronize();
        cudaCheckError();
        dirty = false;
    }

    if (imageInputTexture!=0) {
        ImGui::Image((ImTextureID)imageInputTexture, ImVec2(imageInputDimensions.width, imageInputDimensions.height));
    }
    else {
        ImGui::Text("No input data to show here.");
    }

    if (imageOutputTexture != 0) {
        ImGui::Image((ImTextureID)imageOutputTexture, ImVec2(imageOutputDimensions.width, imageOutputDimensions.height));
    }
    else {
        ImGui::Text("No output data to show here.");
    }
}