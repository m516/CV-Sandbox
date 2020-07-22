#pragma once
#include <string>
#include "generic_cpu_raytracer.h"
#include "raytracing_utils.h"
#include "util/vec3.h"


//Renders a normals-Colored sphere in the middle of the image and 
class CPURaytracer1 : GenericCPUTracer{
public:
	CPURaytracer1() {
		shortDescription = "CPU Raytracer 01: A Colorful rectangle";
		longDescription = "This simple algorithm doesn't actually do any Raytracing. \n\nHowever, due to its simplicity, it does make a good test for the UI.";
	}
	virtual void addCustomUI();
	virtual Color colorAt(float x, float y, float aspectRatio);
};