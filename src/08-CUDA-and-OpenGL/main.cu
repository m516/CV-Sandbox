
#include <iostream>
#include <string>
#include <sstream>
#include <glad.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include "cuda_check_error.h"
#include "timer.h"


#define THREADS_PER_BLOCK 256

int windowWidth = 800, windowHeight = 480;

using namespace std;

__global__ void update_surface(cudaSurfaceObject_t surface, int windowWidth, int windowHeight, int i)
{
  for(int i = 0; i < 1000; i++) ;

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  uint8_t xPrime = (uint8_t) (x + 3*i);
  uint8_t yPrime = (uint8_t) (y + i);

  if(x >= windowWidth)
    return;

  uchar4 pixel = { xPrime & 0xff, yPrime & 0xff, yPrime & 0xff, 0xff };

  surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
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
    // Create the texture
    unsigned char* data = new unsigned char[windowWidth*windowHeight*4]; 
    glTexImage2D(GL_TEXTURE_2D,     // Type of texture
        0,                 // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGBA,            // Internal colour format to convert to
        windowWidth,          // Image width  i.e. 640 for Kinect in standard mode
        windowHeight,          // Image height i.e. 480 for Kinect in standard mode
        0,                 // Border width in pixels (can either be 1 or 0)
        GL_RGBA, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE,  // Image data type
        data);        // The actual image data itself

    //Create the CUDA array and graphics resource
    cudaArray *bitmap_d;
    cudaGraphicsResource *cudaTextureID;
    
    cudaGraphicsGLRegisterImage(&cudaTextureID, textureID, GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlagsNone);
    cudaCheckError();
  
    cudaGraphicsMapResources(1, &cudaTextureID, 0);
    cudaCheckError();
  
    cudaGraphicsSubResourceGetMappedArray(&bitmap_d, cudaTextureID, 0, 0);
    cudaCheckError();
  
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
  
    resDesc.res.array.array = bitmap_d;
    cudaSurfaceObject_t bitmap_surface = 0;
    cudaCreateSurfaceObject(&bitmap_surface, &resDesc);
    cudaCheckError();
  
    dim3 blocks(ceil((float)windowWidth / THREADS_PER_BLOCK), windowHeight);

    //Frame counter
    int i = 0;

    while (!glfwWindowShouldClose(window)) {
      //Start the timer
      Timer stopwatch;
      //Update the texture with the CUDA kernel
      update_surface<<<blocks, THREADS_PER_BLOCK>>>(bitmap_surface, windowWidth, windowHeight, i++);
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
