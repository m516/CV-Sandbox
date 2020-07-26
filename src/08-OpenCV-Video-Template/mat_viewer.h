#pragma once

#include <opencv2/opencv.hpp>

#include <glad.h>
#include <GLFW/glfw3.h>

using namespace cv;
using namespace std;

class MatViewer {
public:
    /*Creates a new MatViewer that binds to a Mat*/
    MatViewer(const char* name, Mat& mat);
    /*Destroys a MatViewer*/
    ~MatViewer();
    /*
    Uses ImGui to render an image preview.

    The first parameter, withControls, adds display controls like the apparent size of the image
    on the display.

    The second parameter, withTooltip, adds a tooltip that magnifies a part of the image when hovered
    */
    void addToGUI(bool withControls = true, bool withTooltip = true);
    /*Gets a reference to the current Mat that this renders*/
    Mat* getMat() { return mat; }
    /*Updates the OpenGL texture*/
    void update();
    /*Gets the dimensions of this Mat and places them in width and height*/
    void getDimensions(int* width, int* height) { *width = this->width; *height = this->height; };
private:
    Mat* mat;
    GLuint textureID = 0;
    const char* name;
    int width = 0, height = 0;

    /*A helper function to create the texture*/
    void generateTexture();
    /*A helper function to reload an existing texture*/
    void reloadTexture();
    /*A helper function to create the texture from a Mat. Returns the texture ID*/
    GLuint matToTexture(const cv::Mat& mat);
};