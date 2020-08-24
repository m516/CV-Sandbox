

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cudaGL.h>
#include <cuda_gl_interop.h>

#include <iostream>



#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
      printf("Cuda error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }

void createTextureSurfacePair(int width, int height, uint8_t* const data, GLuint& textureOut, cudaGraphicsResource_t& graphicsResourceOut, cudaSurfaceObject_t& surfaceOut) {

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
        width,            // Image width  i.e. 640 for Kinect in standard mode
        height,           // Image height i.e. 480 for Kinect in standard mode
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
    cudaArray* bitmap_d;
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


void initGL() {

    // Setup window
    if (!glfwInit())
        return;

    // Decide GL+GLSL versions
#if __APPLE__
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* currentGLFWWindow = glfwCreateWindow(1280, 720, "GLFW Window", NULL, NULL);
    if (currentGLFWWindow == NULL)
        return;
    glfwMakeContextCurrent(currentGLFWWindow);
    glfwSwapInterval(3); // Enable vsync

    if (!gladLoadGL()) {
        // GLAD failed
        printf( "GLAD failed to initialize :(" );
        return;
    }
}


int main() {
    initGL();

    int size = 500;

    uint8_t* data = new uint8_t[size * size * 3]; //dummy 100x100 RGB image

    cudaSurfaceObject_t a;
    cudaGraphicsResource_t b;
    GLuint c;

    for (int i = 0; i < 10000; i++) {
/*------ATTEMPT TO CREATE CUDA SURFACE AND OPENGL TEXTURE------------*/
        createTextureSurfacePair(size, size, data, c, b, a);

/*------ATTEMPT TO DESTROY CUDA SURFACE AND OPENGL TEXTURE------------*/
//https://stackoverflow.com/questions/63455251/how-to-destroy-cuda-graphics-datatypes
        //Destroy surface
        cudaDestroySurfaceObject(a);
        //Destroy graphics resource
        cudaGraphicsUnmapResources(1, &b);
        cudaGraphicsUnregisterResource(b);
        //Destroy texture
        glDeleteTextures(1, &c);

        if (i % 100 == 0) printf("Iteration %d\n", i);
    }
}