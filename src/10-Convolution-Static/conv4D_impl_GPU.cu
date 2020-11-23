#include "conv4D_data_structures.h"
extern "C" {
    #include "conv4D_impl.h"
    #include <math.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <stdio.h>
}



#define cudaCheckError() { \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    printf("Cuda error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
}


__device__ input_feature_map_t gpu_input;
__device__ output_feature_map_t gpu_output;
__device__ conv4d_layer_t gpu_layer;



// function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

//Convolve the arrays with the GPU
__global__
void conv4d_gpu_convolve()
{  
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;

    //printf("Thread ID: (%d,%d,%d)\tBlock ID: (%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

    //Begin convolution
    for (size_t n = 0; n < OUTPUT_BATCHES; n++){
        for (size_t q = blockIdx.x*blockDim.x+threadIdx.x; q < OUTPUT_HEIGHT; q+=gridDim.x * blockDim.x){
            for (size_t p = blockIdx.y*blockDim.y+threadIdx.y; p < OUTPUT_WIDTH; p+=gridDim.y * blockDim.y){
                for (size_t s = 0; s < LAYER_HEIGHT; s++)
                    for (size_t r = 0; r < LAYER_WIDTH; r++)
                        for (size_t m = 0; m < OUTPUT_CHANNELS; m++)
                            for (size_t c = 0; c < INPUT_CHANNELS; c++){
                                gpu_output.data[n][q][p][m] += gpu_input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c] * gpu_layer.weights[s][r][c][m];
                            }
            
                for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
                    gpu_output.data[n][q][p][m] += gpu_layer.bias[m];
                    if(gpu_output.data[n][q][p][m] < 0) gpu_output.data[n][q][p][m] = 0;
                }      
            }
        }
    }
    
}


void cuda_var_update(){
    cudaMemcpyToSymbol(gpu_input, &input, sizeof(input_feature_map_t));
    cudaCheckError();
    cudaMemcpyToSymbol(gpu_layer, &layer, sizeof(conv4d_layer_t));
    cudaCheckError();
}

void host_var_update(){
    cudaMemcpyFromSymbol(&output, gpu_output, sizeof(output_feature_map_t));
    cudaCheckError();
}


void conv4d_convolve_cuda_discrete(int block_size){

    if(block_size<=0) CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("GPU block size expected to be larger than 0. Got %d\n", block_size);

    void* gpu_output_addr = NULL;
    cudaGetSymbolAddress<output_feature_map_t>(&gpu_output_addr, gpu_output);
    cudaCheckError()
    cudaMemset(gpu_output_addr, 0, sizeof(output_feature_map_t));
    cudaCheckError();

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(OUTPUT_HEIGHT / dimBlock.x, OUTPUT_WIDTH / dimBlock.y);

    // dim3 dimBlock(1, 1);
    // dim3 dimGrid(1, 1);
    conv4d_gpu_convolve<<<dimBlock,dimGrid>>>();
    cudaCheckError();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaCheckError();
    host_var_update();
}

void conv4d_convolve_cuda_discrete_rewrite_gpu_data(int block_size){
    cudaCheckError();
    cuda_var_update();
    cudaCheckError();
    conv4d_convolve_cuda_discrete(block_size);
    cudaCheckError();
}

void conv4d_convolve_cuda_CUDNN(){

}