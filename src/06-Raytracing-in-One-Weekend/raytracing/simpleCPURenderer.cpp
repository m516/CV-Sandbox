#include "simpleCPURenderer.h"


void SimpleCPURaytracer::beginRendering(Renderer* r, GenericCPUTracer* t, int x, int y, int w, int h)
{
	// Start thread
	currentThread = new std::thread(&SimpleCPURaytracer::render, this, r, t, x, y, w, h);
}


void SimpleCPURaytracer::cancelRender()
{
	cancelRequested = true;
}

//Render
void SimpleCPURaytracer::render(Renderer* r, GenericCPUTracer* t, int borderX, int borderY, int borderWidth, int borderHeight)
{
	//Delay. This seems to help make the program run more consistently
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	//Begin the timer
	auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	std::cout << "Began rendering at " << std::ctime(&start_time);
	//Set flags. These affect the UI
	isBusy = true;
	isDone = false;
	cancelRequested = false;
	renderProgress = 0.f;
	//Get the size of the render area. 
	//If borderWidth or borderHeight is less than 0, w or h is assigned to the bounds of the image.
	int w, h;
	r->getDimensions(&w, &h);
	w -= borderX;
	h -= borderY;
	//Otherwise, it's set to the border
	if (borderWidth > 0) w = borderWidth;
	if (borderHeight> 0) h = borderHeight;
	//Print the width and height of the render area for debugging
	std::cout << "\tWidth:  " << w;
	std::cout << "\tHeight: " << h;
	std::cout << std::endl;
	
	//Iterate through all rows
	for (int j = borderY; j < h; j++) {
		//Update the render progress here.
		renderProgress = (float)j / (float)h;
		//Iterate through all pixels in row j
		for (int i = borderX; i < w; i++) {
			//Stop rendering if cancelRender() has been called.
			if (cancelRequested) {
				isBusy = false;
				std::cerr << "Render canceled" << std::endl;
				return;
			}
			//Map the position so that (0,0) is the upper left hand corner of the image and
			//(1,1) is the lower right hand corner of the image.
			double x = (double)(i) / (double)(w);
			double y = (double)(j) / (double)(h);
			//Using the CPUTracer "t", evalutate the color at the position (x,y)
			//Place the result at pixel (i,j) in the image renderer "r" so it can be displayed by the UI
			r->setPixel(i, j, t->colorAt(x, y));
		}
	}
	//Grab the time when rendering was complete
	auto end = std::chrono::system_clock::now();
	//Calculate the amount of time that elapsed between the start time and end time
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	renderTime = elapsed_seconds.count();
	//Print the result
	std::cout << "Rendering time: " << renderTime << std::endl;

	//Rendering was successful. Set flags to let the UI know the operation was successful
	isBusy = false;
	isDone = true;
}

