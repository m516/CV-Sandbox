#pragma once

#include "mat_viewer.h"
#include <imgui/imgui.h>

class VideoViewer{
public:
    /*
    Binds this VideoViewer to an existing VideoCapture instance.
    */
    VideoViewer(VideoCapture& videoCapture);

    /*
    Uses ImGui to render an image preview.

    The first parameter, manualPlayback, adds human-readable controls for the playback of the 
    video, including advancing to the next frame and scrubbing through a video feed.

    The second parameter, imageControls, adds display controls like the apparent size of the image
    on the display.
    */
    void addToGUI(bool manualPlayback = true, bool imageControls = true);

    /*Returns the index of the current frame*/
    int getCurrentFrame();
    /*Sets the index of the current frame*/
    void setCurrentFrame(int desiredFrame);
    /*Attempts to advance to the next frame. This is a blocking function and may take time*/
    bool nextFrame();
    /*Returns the number of accessible frames in the video feed*/
    int numFrames();
    /*The current OpenCV image*/
    Mat mat;
    /*The address of the current video feed*/
    VideoCapture* video;
private:
    /*This class relies on MatViewer to render the matrix*/
    MatViewer* matViewer;
    /*True if playing*/
    bool isPlaying = false;
};