#include "dummy_cuda_algorithm.cuh"

#include <stdio.h>
#include <assert.h>
#include <iostream>


#define THREADS_PER_BLOCK 128


__global__ void kernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int width, int height) {

	//Get the pixel index
	unsigned int xPx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int yPx = threadIdx.y + blockIdx.y * blockDim.y;


	//Don't do any computation if this thread is outside of the surface bounds.
	if (xPx >= width || yPx >= height) return;

	//Copy the contents of input to output.
	uchar4 pixel = {255,128,0,255};
	//Read a pixel from the input. Disable to default to the flat orange color above
	surf2Dread<uchar4>(&pixel, input, xPx * sizeof(uchar4), yPx, cudaBoundaryModeClamp);
	surf2Dwrite(pixel, output, xPx * sizeof(uchar4), yPx);
}

bool DummyCUDAAlgorithm::process()
{
	//Don't process empty data
	if (invalid()) return false;
	
	//Call the algorithm

	//Set the number of blocks to call the kernel with.
	dim3 blocks((unsigned int)ceil((float)imageInputDimensions.width / THREADS_PER_BLOCK), imageInputDimensions.height);
	kernel<<<blocks, THREADS_PER_BLOCK>>>(d_imageInputTexture, d_imageOutputTexture, imageInputDimensions.width, imageInputDimensions.height);

	return CUDAVisionAlgorithm::process();
}
