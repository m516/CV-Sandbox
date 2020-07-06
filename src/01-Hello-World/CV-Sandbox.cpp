// CV-Sandbox.cpp : Defines the entry point for the application.
//

#include "CV-Sandbox.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	// Print out a message of elation, now that OpenCV is installed!
	cout << "Hello OpenCV!" << endl;

	// Get an image name 
	string filename = MEDIA_DIRECTORY;
	// MEDIA_DIRECTORY is defined in the root-level CMake script
	filename += "RedFox.png";

	// Load the image
	Mat image = imread(filename);
	
	// Show that image
	imshow("Test image", image);

	// Wait until the user is finished seeing the image.
	waitKey(0);

	//Now we're done.
	return 0;
}
