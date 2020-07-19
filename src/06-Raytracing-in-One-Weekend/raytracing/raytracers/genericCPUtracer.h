#pragma once
#include "../raytracingutils.h"
#include "imgui/imgui.h"

class GenericCPUTracer {
public:
	/*
	Computes the color at a certain location in the image, where (0,0) is the upper left-hand corner of the image, and (1,1) is the lower right-hand corner. 
	This is the only function that must be implemented by a subclass
	
	*/
	virtual color colorAt(float x, float y)=0;
	/*
	Allows a renderer to add custom settings directly to the UI
	*/
	virtual void addCustomUI() { ImGui::Text("This renderer doesn't have any configurable options"); }
	const char* shortDescription = "Generic CPU raytracer";
	const char* longDescription = "This CPU raytracer doesn't have a long description yet.";
};