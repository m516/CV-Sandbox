#pragma once

#include <glad/glad.h>
#include "Raytracing/Raytracingutils.h"

class Renderer {
public:
	Renderer();
	~Renderer();
	GLuint getTextureID();
	unsigned char* getRawImageData();
	bool isInitialized() { return initialized; }
	void getDimensions(int* widthPtr, int* heightPtr) { *widthPtr = textureWidth, * heightPtr = textureHeight; }
	void loadTexture(int width, int height);
	void reloadTexture();
	void setPixel(int x, int y, Color c);
private:
	volatile bool initialized = false;
	unsigned char* imageData = nullptr;
	GLuint texture = 0;
	int textureWidth = 0, textureHeight = 0;
	void clearData();
};