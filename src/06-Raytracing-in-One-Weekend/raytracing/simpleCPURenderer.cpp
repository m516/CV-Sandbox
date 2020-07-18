#include "simpleCPURenderer.h"

void SimpleCPURaytracer::beginRendering(Renderer* r, int x, int y, int w, int h)
{
	// Start thread
	currentThread = new std::thread(&SimpleCPURaytracer::render, this, r, x, y, w, h);
}

void SimpleCPURaytracer::cancelRender()
{
	cancelRequested = true;
}

void SimpleCPURaytracer::render(Renderer* r, int borderX, int borderY, int borderWidth, int borderHeight)
{

	std::this_thread::sleep_for(std::chrono::milliseconds(10));

	auto start = std::chrono::system_clock::now();
	std::time_t start_time = std::chrono::system_clock::to_time_t(start);
	std::cout << "Began rendering at " << std::ctime(&start_time);

	isBusy = true;
	isDone = false;
	cancelRequested = false;
	renderProgress = 0.f;

	int w, h;
	r->getDimensions(&w, &h);
	if (borderWidth > 0) w = borderWidth;
	if (borderHeight> 0) h = borderHeight;

	std::cout << "\tWidth:  " << w;
	std::cout << "\tHeight: " << h;
	std::cout << std::endl;
	
	for (int i = borderX; i < w; i++) {
		renderProgress = (float)i / (float)w;
		for (int j = borderY; j < h; j++) {
			if (cancelRequested) {
				isBusy = false;
				return;
			}

			double x = (double)(i) / (double)(w);
			double y = (double)(j) / (double)(h);
			color c(x, y, 1.-y);
			r->setPixel(i, j, c);
		}
	}

	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	renderTime = elapsed_seconds.count();
	std::cout << "Rendering time: " << renderTime;

	isBusy = false;
	isDone = true;
}
