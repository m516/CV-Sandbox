
#include <iostream>
#include <string>
#include <sstream>
#include <glad.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include "cuda_check_error.h"
#include "timer.h"


#define THREADS_PER_BLOCK 128

int windowWidth = 800, windowHeight = 480;

using namespace std;

//A custom, lightweight CUDA implementation of a 2D vector with floats.
class CUDAVec2{
public:
  __device__ CUDAVec2(float x = 0, float y = 0){this->x = x; this->y=y;}
  __device__ CUDAVec2(CUDAVec2& v){x = v.x; y=v.y;}
  float x, y;
  //Get the magnitude of this vector.
  __device__ float mag(){return hypotf(x,y);}
  //Subtract an amount from this vector.
  __device__ void subtract(CUDAVec2 amount){x-=amount.x;y-=amount.y;}
  //Subtract an amount from this vector.
  __device__ void subtract(float x, float y){this->x-=x;this->y-=y;}
};

__global__ void update_surface(cudaSurfaceObject_t surface, int textureWidth, int textureHeight, int i)
{
  int xPx = threadIdx.x + blockIdx.x * blockDim.x;
  int yPx = threadIdx.y + blockIdx.y * blockDim.y;

  if(xPx >= textureWidth)
    return;
  if(yPx >= textureHeight)
    return;

  CUDAVec2 normalizedCoordinates((float)xPx / textureWidth, (float)yPx / textureHeight);
  i%=400;
  float theta = (float)i/63.661977236758134307553505349006f;

  bool inCircle = false;

  //Create and find the intersection of a bunch of circles
  int numCircles = 10;
  for(int i = 0; i < numCircles; i++){
    theta+=(float)6.283185307179586476925286766559f/numCircles;
    CUDAVec2 circlePosition(0.5+0.4*cos(theta), 0.5+0.4*sin(theta));
    CUDAVec2 v (normalizedCoordinates);
    v.subtract(circlePosition);
    float r = v.mag();
    if(r<0.1){
      inCircle = true;
      break;
    }
  }


  float red = normalizedCoordinates.x;
  float green = 1.f - normalizedCoordinates.y;
  float blue = 1.f-normalizedCoordinates.x;
  float alpha = 1.f;

  if(inCircle){
    red = 1.f - red;
    green = 1.f - green;
    blue = 1.f - blue;
  }


  uchar4 pixel = { (uint8_t)(red*255),
    (uint8_t)(green*255),
    (uint8_t)(blue*255),
    (uint8_t)(alpha*255)};

  surf2Dwrite(pixel, surface, xPx * sizeof(uchar4), yPx);
}

static void init_opengl(int w, int h) {
  glViewport(0, 0, w, h); // use a screen size of windowWidth x windowHeight

  glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
  glLoadIdentity();
  glOrtho(0.0, w, h, 0.0, 0.0, 100.0);

  glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
}

/**
 * A helper function for terminating the program
 */
void terminate(int errorCode) {
    cout << "Closing application";
    //Close GLFW
    glfwTerminate();
    //Exit
    exit(errorCode);
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

static void resize_callback(GLFWwindow* window, int new_width, int new_height) {
  glViewport(0, 0, windowWidth = new_width, windowHeight = new_height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, windowWidth, windowHeight, 0.0, 0.0, 100.0);
  glMatrixMode(GL_MODELVIEW);
}

static void error_callback(int error, const char* description) {
  fprintf(stderr, "Error: %s\n", description);
}

int main(int argc, char **argv)
{
    // Initialize GLFW, and GLAD, in exactly the same way as project 4.
    GLFWwindow* window;
    glfwSetErrorCallback(error_callback);
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    window = glfwCreateWindow(windowWidth, windowHeight, "Simple CUDA + GLFW example", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwSetKeyCallback(window, key_callback);
    glfwSetWindowSizeCallback(window, resize_callback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGL()) {
        cerr << "GLAD failed to initialize :(";
        terminate(1);
    }
    init_opengl(windowWidth, windowHeight);
    // End GLAD and GLFW setup

    // Create the OpenGL texture that will be displayed with GLAD and GLFW
    GLuint textureID;
    glGenTextures(1, &textureID);   
    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);
    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    // Create the texture and its attributes
    int textureWidth = windowWidth, textureHeight = windowHeight;
    unsigned char* data = new unsigned char[textureWidth*textureHeight*4]; 
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
        0,                 // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGBA,            // Internal colour format to convert to
        windowWidth,          // Image width  i.e. 640 for Kinect in standard mode
        windowHeight,          // Image height i.e. 480 for Kinect in standard mode
        0,                 // Border width in pixels (can either be 1 or 0)
        GL_RGBA, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE,  // Image data type
        data);        // The actual image data itself

    //Create the CUDA array and texture reference
    cudaArray *bitmap_d;
    cudaGraphicsResource *cudaTextureID;
    //Register the GL texture with the CUDA graphics library. A new cudaGraphicsResource is created, and its address is placed in cudaTextureID.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g80d12187ae7590807c7676697d9fe03d
    cudaGraphicsGLRegisterImage(&cudaTextureID, textureID, GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlagsNone);
    cudaCheckError();
    //Map graphics resources for access by CUDA.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gad8fbe74d02adefb8e7efb4971ee6322
    cudaGraphicsMapResources(1, &cudaTextureID, 0);
    cudaCheckError();
    //Get the location of the array of pixels that was mapped by the previous function and place that address in bitmap_d
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
    cudaGraphicsSubResourceGetMappedArray(&bitmap_d, cudaTextureID, 0, 0);
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
    cudaSurfaceObject_t bitmap_surface = 0;
    //Create the surface with the given description. That surface ID is placed in bitmap_surface.
    //Documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT_1g958899474ab2c5f40d233b524d6c5a01
    cudaCreateSurfaceObject(&bitmap_surface, &resDesc);
    cudaCheckError();
  
    dim3 blocks((unsigned int)ceil((float)textureWidth / THREADS_PER_BLOCK), textureHeight);

    //Frame counter
    int i = 0;

    while (!glfwWindowShouldClose(window)) {
      //Start the timer
      Timer stopwatch;
      //Update the texture with the CUDA kernel
      update_surface<<<blocks, THREADS_PER_BLOCK>>>(bitmap_surface, textureWidth, textureHeight, i++);
      //Print elapsed time occasionally
      if(i%10==0){
        float fps = 1.f/stopwatch.getElapsedSeconds();
        ostringstream myString;
        myString << "Simple CUDA + GLFW example (";
        myString.precision(2);
        myString << std::fixed << fps;
        myString << " FPS)";
        glfwSetWindowTitle(window, myString.str().c_str());
      }
      cudaCheckError();
      cudaDeviceSynchronize();
      cudaCheckError();

      //Render the results on the screen buffer
      //Clear color and depth buffers
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      //Begin drawing
      glMatrixMode(GL_MODELVIEW);     //Operate on model-view matrix
      glEnable(GL_TEXTURE_2D);
      //Bind to our texture handle
      glBindTexture(GL_TEXTURE_2D, textureID);
      //Draw a quad
      glBegin(GL_QUADS);
      glTexCoord2i(0, 0); glVertex2i(0, 0);
      glTexCoord2i(0, 1); glVertex2i(0, windowHeight);
      glTexCoord2i(1, 1); glVertex2i(windowWidth, windowHeight);
      glTexCoord2i(1, 0); glVertex2i(windowWidth, 0);
      glEnd();
      glDisable(GL_TEXTURE_2D);
      
      //Blit rendered contents on the screen and poll events.
      glfwSwapBuffers(window);
      glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    exit(EXIT_SUCCESS);
  


  return 0;
}
