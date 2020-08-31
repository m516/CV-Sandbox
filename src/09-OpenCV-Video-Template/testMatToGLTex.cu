

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cudaGL.h>
#include <cuda_gl_interop.h>

#include <iostream>


/** Macro for checking if CUDA has problems */
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
      printf("Cuda error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }


/*Window dimensions*/
const int windowWidth = 1280, windowHeight = 720;
/*Window address*/
GLFWwindow* currentGLFWWindow = 0;


/**
 * A simple image processing kernel that copies the inverted data from the input surface to the output surface.
 */
__global__ void kernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int width, int height) {

    //Get the pixel index
    unsigned int xPx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int yPx = threadIdx.y + blockIdx.y * blockDim.y;


    //Don't do any computation if this thread is outside of the surface bounds.
    if (xPx >= width || yPx >= height) return;

    //Copy the contents of input to output.
    uchar4 pixel = { 255,128,0,255 };
    //Read a pixel from the input. Disable to default to the flat orange color above
    surf2Dread<uchar4>(&pixel, input, xPx * sizeof(uchar4), yPx, cudaBoundaryModeClamp);

    //Invert the color
    pixel.x = ~pixel.x;
    pixel.y = ~pixel.y;
    pixel.z = ~pixel.z;

    //Write the new pixel color to the 
    surf2Dwrite(pixel, output, xPx * sizeof(uchar4), yPx);
}

class Processor {
public:
    void setInput( uint8_t* const data, int imageWidth, int imageHeight);
    void processData();
    GLuint getInputTexture();
    GLuint getOutputTexture();
    void writeOutputTo(uint8_t* destination);
private:
    /**
    * @brief True if the textures and surfaces are initialized.
    *
    * Prevents memory leaks
    */
    bool surfacesInitialized = false;
    /**
     * @brief The width and height of a texture/surface pair.
     *
     */
    struct ImgDim { int width, height; };
    /**
     * @brief Creates a CUDA surface object, CUDA resource, and OpenGL texture from some data.
     */
    void createTextureSurfacePair(const ImgDim& dimensions, uint8_t* const data, GLuint& textureOut, cudaGraphicsResource_t& graphicsResourceOut, cudaSurfaceObject_t& surfaceOut);
    /**
     * @brief Destroys every CUDA surface object, CUDA resource, and OpenGL texture created by this instance.
     */
    void destroyEverything();
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
    /** Returns true if nothing can be rendered */
    bool empty() { return imageInputTexture == 0; }

};


void Processor::setInput(uint8_t* const data, int imageWidth, int imageHeight)
{


    //Same-size images don't need texture regeneration, so skip that.
    if (imageHeight == imageInputDimensions.height && imageWidth == imageInputDimensions.width) {


        /*
        Possible shortcut: we know the input is the same size as the texture and CUDA surface object.
        So instead of destroying the surface and texture, why not just overwrite them?

        That's what I try to do in the following block, but because "data" is BGR and the texture
        is RGBA, the channels get all messed up.
        */

        /*
        //Use the input surface's CUDAResourceDesc to gain access to the surface data array
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        cudaGetSurfaceObjectResourceDesc(&resDesc, d_imageInputTexture);
        cudaCheckError();

        //Copy the data from the input array to the surface
        cudaMemcpyToArray(resDesc.res.array.array, 0, 0, input.data, imageInputDimensions.width * imageInputDimensions.height * 3, cudaMemcpyHostToDevice);
        cudaCheckError();

        //Set status flags
        surfacesInitialized = true;

        return;
        */
    }


    //Clear everything that originally existed in the texture/surface
    destroyEverything();

    //Get the size of the image and place it here.
    imageInputDimensions.width = imageWidth;
    imageInputDimensions.height = imageHeight;
    imageOutputDimensions.width = imageWidth;
    imageOutputDimensions.height = imageHeight;

    //Create the input surface/texture pair
    createTextureSurfacePair(imageInputDimensions, data, imageInputTexture, d_imageInputGraphicsResource, d_imageInputTexture);

    //Create the output surface/texture pair
    uint8_t* outData = new uint8_t[imageOutputDimensions.width * imageOutputDimensions.height * 3];
    createTextureSurfacePair(imageOutputDimensions, outData, imageOutputTexture, d_imageOutputGraphicsResource, d_imageOutputTexture);
    delete outData;

    //Set status flags
    surfacesInitialized = true;
}

void Processor::processData()
{
    const int threadsPerBlock = 128;

    //Call the algorithm

    //Set the number of blocks to call the kernel with.
    dim3 blocks((unsigned int)ceil((float)imageInputDimensions.width / threadsPerBlock), imageInputDimensions.height);
    kernel <<<blocks, threadsPerBlock >>> (d_imageInputTexture, d_imageOutputTexture, imageInputDimensions.width, imageInputDimensions.height);

    //Sync the surface with the texture
    cudaDeviceSynchronize();
    cudaCheckError();
}

GLuint Processor::getInputTexture()
{
    return imageInputTexture;
}

GLuint Processor::getOutputTexture()
{
    return imageOutputTexture;
}

void Processor::writeOutputTo(uint8_t* destination)
{
    //Haven't figured this out yet
}

void Processor::createTextureSurfacePair(const Processor::ImgDim& dimensions, uint8_t* const data, GLuint& textureOut, cudaGraphicsResource_t& graphicsResourceOut, cudaSurfaceObject_t& surfaceOut) {

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

void Processor::destroyEverything()
{
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


/** A way to initialize OpenGL with GLFW and GLAD */
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
    currentGLFWWindow = glfwCreateWindow(windowWidth, windowHeight, "Output image (OpenGL + GLFW)", NULL, NULL);
    if (currentGLFWWindow == NULL)
        return;
    glfwMakeContextCurrent(currentGLFWWindow);
    glfwSwapInterval(3); // Enable vsync

    if (!gladLoadGL()) {
        // GLAD failed
        printf( "GLAD failed to initialize :(" );
        return;
    }

    //Change GL settings
    glViewport(0, 0, windowWidth, windowHeight); // use a screen size of WIDTH x HEIGHT

    glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
    glLoadIdentity();
    glOrtho(0.0, windowWidth, windowHeight, 0.0, 0.0, 100.0);

    glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
}

/** Renders the textures on the GLFW window and requests GLFW to update */
void showTextures(GLuint top, GLuint bottom) {
    // Clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);     // Operate on model-view matrix

    glBindTexture(GL_TEXTURE_2D, top);
    /* Draw top quad */
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0, 0);
    glTexCoord2i(0, 1); glVertex2i(0, windowHeight/2);
    glTexCoord2i(1, 1); glVertex2i(windowWidth, windowHeight / 2);
    glTexCoord2i(1, 0); glVertex2i(windowWidth, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    /* Draw top quad */
    glBindTexture(GL_TEXTURE_2D, bottom);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0, windowHeight / 2);
    glTexCoord2i(0, 1); glVertex2i(0, windowHeight);
    glTexCoord2i(1, 1); glVertex2i(windowWidth, windowHeight);
    glTexCoord2i(1, 0); glVertex2i(windowWidth, windowHeight / 2);
    glEnd();
    glDisable(GL_TEXTURE_2D);


    glfwSwapBuffers(currentGLFWWindow);
    glfwPollEvents();
}


int main() {
    initGL();

    int imageWidth = windowWidth;
    int imageHeight = windowHeight / 2;

    uint8_t* imageData = new uint8_t[imageWidth * imageHeight * 3];

    Processor p;

    while (!glfwWindowShouldClose(currentGLFWWindow))
    {
        //Process the image here
        p.setInput(imageData, imageWidth, imageHeight);
        p.processData();
        showTextures(p.getInputTexture(), p.getOutputTexture());
    }
}

