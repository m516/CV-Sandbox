#pragma once

#include <thread>
#include <mutex>
#include <ctime> 

#include "Raytracingutils.h"
#include "Raytracers/Raytracers.h"
#include "../renderer.h"


class SimpleCPURaytracer {
public:
	/*
	Starts the rendering process and shoves it into another thread, freeing the current one for graphics

	Parameters:
		r - a reference to an object that stores the image and possibly shows it on the UI

		t - a reference to a CPURaytracer that can evaluate the Color of an arbitrary pixel

		x, y, w, h - the dimensions of the rectangle on the image that needs to be rendered (in pixel coordinates).
		This rectangle's upper left-hand corner is at the position (x,y), and extends w pixels wide by h pixels tall.
		This means the lower-rightmost corner pixel that's rendered is at (x+w-1,y+h-1)

	If only the first two arguments are provided, the entire image will be rendered.
	*/
	void beginRendering(Renderer* r, GenericCPUTracer* t, int x = 0, int y = 0, int w = -1, int h = -1);


	/*
	Cancels a running render thread from outside of that thread. Ideal for a "cancel button" on a UI
	*/
	void cancelRender();


	/*
	Starts the rendering process ON THE SAME THREAD, blocking the current thread until rendering is complete

	Parameters:
		r - a reference to an object that stores the image and possibly shows it on the UI

		t - a reference to a CPURaytracer that can evaluate the Color of an arbitrary pixel

		x, y, w, h - the dimensions of the rectangle on the image that needs to be rendered (in pixel coordinates).
		This rectangle's upper left-hand corner is at the position (x,y), and extends w pixels wide by h pixels tall.
		This means the lower-rightmost corner pixel that's rendered is at (x+w-1,y+h-1)

	If only the first two arguments are provided, the entire image will be rendered.
	*/
	void render(Renderer* r, GenericCPUTracer* t, int x = 0, int y = 0, int w = -1, int h = -1);


	/*Returns true if a render operation is in progress*/
	bool busy() { return isBusy; }
	/*Returns true if a render has successfully completed*/
	bool done() { return isDone; }
	/*Returns how far the render has progressed, a value between 0 and 1*/
	float progress() { return renderProgress; }
	/*Returns the time taken to render the most recent render*/
	float getRenderTime() { return renderTime; }
private:
	bool isBusy = false,
		isDone = false;
	bool cancelRequested = false;
	float renderProgress = 0.f;
	float renderTime = 0.f;
	std::thread* currentThread = nullptr;
};