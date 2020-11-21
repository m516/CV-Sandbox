//#include "HelloCUDA.h"

#include <iostream>
#include <math.h>
#include "conv4D_impl.h"


#ifdef CUDA_SUPPORT
conv_ret conv4d_convolve_cuda_discrete(conv4d_layer layer, featuremap_3d input, featuremap_3d output){
    //Benchmarking setup
    conv_ret r;
    time_t start_t, end_t;
    time(&start_t);
    //TODO stub
    //End benchmarking         
    time(&end_t);
    r.time_elapsed = difftime(end_t, start_t);
    return r;
}
conv_ret conv4d_convolve_cuda_CUDNN(conv4d_layer layer, featuremap_3d input, featuremap_3d output){
    //Benchmarking setup
    conv_ret r;
    time_t start_t, end_t;
    time(&start_t);
    //TODO stub
    //End benchmarking         
    time(&end_t);
    r.time_elapsed = difftime(end_t, start_t);
    return r;
}
#endif

/*
// function to add the elements of two arrays
__global__
void doStuff(int n, float* x, float* y)
{  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20; // 1M elements

    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
*/