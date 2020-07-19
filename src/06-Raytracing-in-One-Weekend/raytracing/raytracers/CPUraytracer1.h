#pragma once
#include <string>
#include "genericCPUtracer.h"

//Renders a normals-colored sphere in the middle of the image and 
class CPURaytracer1 : GenericCPUTracer{
public:
	CPURaytracer1() {
		shortDescription = "CPU Raytracer 01: A colorful rectangle";
		longDescription = "This simple algorithm doesn't actually do any raytracing. \n\nHowever, due to its simplicity, it does make a good test for the UI.";
	}
	virtual void addCustomUI();
	virtual color colorAt(float x, float y);
};