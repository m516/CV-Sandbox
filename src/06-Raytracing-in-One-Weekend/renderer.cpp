#include "renderer.h"

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
    clearData();
}

GLuint Renderer::getTextureID()
{
    return texture;
}

unsigned char* Renderer::getRawImageData()
{
    return imageData;
}

void Renderer::clearData()
{
    glDeleteTextures(1, &texture);
    delete[] imageData;
    textureWidth = 0;
    textureHeight = 0;
    initialized = false;
}

void Renderer::loadTexture (int width, int height)
{
    if (initialized) clearData();


	textureWidth = width;
	textureHeight = height;

	imageData = new unsigned char[(size_t)width * height * 4];

	// Generate a number for our textureID's unique handle
	glGenTextures(1, &texture);

	// Bind to our texture handle
	glBindTexture(GL_TEXTURE_2D, texture);

	// Set texture interpolation methods for minification and magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Set texture clamping method
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Set incoming texture format to:
	// GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
	// GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
	// Work out other mappings as required ( there's a list in comments in main() )
	GLenum inputColourFormat = GL_RGBA;

	// Create the texture
	glTexImage2D(GL_TEXTURE_2D,     // Type of texture
		0,                 // Pyramid level (for mip-mapping) - 0 is the top level
		GL_RGBA,            // Internal colour format to convert to
		width,             // Image width  i.e. 640 for Kinect in standard mode
		height,            // Image height i.e. 480 for Kinect in standard mode
		0,                 // Border width in pixels (can either be 1 or 0)
		inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
		GL_UNSIGNED_BYTE,  // Image data type
		imageData);        // The actual image data itself

		// If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher
	
		//glGenerateMipmap(GL_TEXTURE_2D);

		initialized = true;
}

void Renderer::reloadTexture()
{
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexSubImage2D(
		GL_TEXTURE_2D,
		0,
		0,
		0,
		textureWidth,
		textureHeight,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		imageData
	);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::setPixel(int x, int y, color c)
{    
	int pos = y * textureWidth + x;
	pos <<= 2;
	imageData[pos] = (unsigned char)(c.x() * 255);
	imageData[pos+1] = (unsigned char)(c.y() * 255);
	imageData[pos+2] = (unsigned char)(c.z() * 255);
	imageData[pos+3] = 255;
}

