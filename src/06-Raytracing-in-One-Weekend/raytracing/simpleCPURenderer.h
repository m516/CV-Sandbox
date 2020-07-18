#pragma once

#include <thread>
#include <mutex>
#include <ctime> 

#include "raytracingutils.h"
#include "../renderer.h"


class SimpleCPURaytracer {
public:
	void beginRendering(Renderer* r, int x = 0, int y = 0, int w = -1, int h = -1);
	void cancelRender();
	void render(Renderer* r, int x = 0, int y = 0, int w = -1, int h = -1);
	bool busy() { return isBusy; }
	bool done() { return isDone; }
	float progress() { return renderProgress; }
	float getRenderTime() { return renderTime; }
private:
	bool isBusy = false,
		isDone = false;
	bool cancelRequested = false;
	float renderProgress = 0.f;
	float renderTime = 0.f;
	std::thread* currentThread = nullptr;
};