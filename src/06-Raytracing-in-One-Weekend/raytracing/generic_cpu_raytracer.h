#pragma once
#include "imgui/imgui.h"
#include "util/vec3.h"

class GenericCPUTracer {
public:
	/*
	Computes the Color at a certain location in the image, where (0,0) is the upper left-hand corner of the image, and (1,1) is the lower right-hand corner. 
	This is the only function that must be implemented by a subclass
	
	*/
	virtual Color colorAt(float x, float y, float aspectRatio)=0;
	/*
	Allows a renderer to add custom settings directly to the UI
	*/
	virtual void addCustomUI() { ImGui::Text("This renderer doesn't have any configurable options"); }
	/*
	Called immediately before any call to "colorAt()".
	It be used to initialize variables that don't need to be calculated every pixel, like a camera matrix.
	It also be used to initialize variables that many pixels depend on, like a brightness counter for dithering.
	*/
	virtual void init() {};
	const char* shortDescription = "Generic CPU Raytracer";
	const char* longDescription = "This CPU Raytracer doesn't have a long description yet.";
};