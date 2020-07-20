#pragma once
#include <string>
#include "genericCPUtracer.h"
#include "../Raytracingutils.h"

//Renders a normals-Colored sphere in the middle of the image and 
class CPURaytracer2 : GenericCPUTracer{
public:
	CPURaytracer2() {
		shortDescription = "CPU Raytracer 02: A Circle";
		longDescription = "This simple algorithm tests Rays hitting spheres.";
	}
	virtual void init();
	virtual void addCustomUI();
	virtual Color colorAt(float x, float y, float aspectRatio);
private:
    // Camera
	Camera camera;
};