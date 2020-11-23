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

    size_t q_init = blockIdx.x*blockDim.x+threadIdx.x;
    size_t p_init = blockIdx.y*blockDim.y+threadIdx.y;
    size_t m_init = blockIdx.z*blockDim.z+threadIdx.z;
    size_t q_stride = gridDim.x * blockDim.x;
    size_t p_stride = gridDim.y * blockDim.y;
    size_t m_stride = gridDim.z * blockDim.z;


    //Begin convolution
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = q_init; q < OUTPUT_HEIGHT; q+=q_stride)
            for (size_t p = p_init; p < OUTPUT_WIDTH; p+=p_stride){
                for (size_t s = 0; s < LAYER_HEIGHT; s++)
                    for (size_t r = 0; r < LAYER_WIDTH; r++)
                        for (size_t c = 0; c < INPUT_CHANNELS; c++)
                            for (size_t m = m_init; m < OUTPUT_CHANNELS; m+=m_stride){
                                gpu_output.data[n][q][p][m] += gpu_input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c] * gpu_layer.weights[s][r][c][m];
                            }
                    for (size_t m = m_init; m < OUTPUT_CHANNELS; m+=m_stride){
                        gpu_output.data[n][q][p][m] += gpu_layer.bias[m];
                        if(gpu_output.data[n][q][p][m] < 0) gpu_output.data[n][q][p][m] = 0;
                    }
                }
}

/**
 * @brief Updates the GPU versions of the input and layer from their corresponding CPU versions
 * 
 */
void cuda_var_update(){
    cudaMemcpyToSymbol(gpu_input, &input, sizeof(input_feature_map_t), 0, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpyToSymbol(gpu_layer, &layer, sizeof(conv4d_layer_t), 0, cudaMemcpyHostToDevice);
    cudaCheckError();
}

/**
 * @brief Updates the CPU version of the output from its corresponding GPU version
 * 
 */
void host_var_update(){
    cudaMemcpyFromSymbol(&output, gpu_output, sizeof(output_feature_map_t), 0, cudaMemcpyDeviceToHost);
    cudaCheckError();
}

/**
 * @brief 
 * 
 * @param block_size 
 */
void conv4d_convolve_cuda_discrete(int block_size, int grid_size){

    if(block_size<=0) CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("GPU block size expected to be larger than 0. Got %d\n", block_size);
    if(grid_size<=0) CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("GPU grid size expected to be larger than 0. Got %d\n", grid_size);

    void* gpu_output_addr;
    cudaGetSymbolAddress<output_feature_map_t>(&gpu_output_addr, gpu_output);
    cudaCheckError()
    cudaMemset(gpu_output_addr, 0, sizeof(output_feature_map_t));
    cudaCheckError();

    dim3 dimBlock(block_size, block_size, block_size);
    dim3 dimGrid(grid_size, grid_size, grid_size);

    cudaCheckError();

    // dim3 dimBlock(1, 1);
    // dim3 dimGrid(1, 1);
    conv4d_gpu_convolve<<<dimGrid,dimBlock>>>();
    cudaCheckError();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaCheckError();
    host_var_update();


    //Reset memory
    // memset(&output, 0, sizeof(output));

    // for(int bx = 0; bx < grid_size; bx++){
    //     for(int by = 0; by < grid_size; by++){
    //         for(int bz = 0; bz < grid_size; bz++){
    //             for(int tx = 0; tx < block_size; tx++){
    //                 for(int ty = 0; ty < block_size; ty++){
    //                     for(int tz = 0; tz < block_size; tz++){
    //                         size_t q_init = bx*block_size+tx;
    //                         size_t p_init = by*block_size+ty;
    //                         size_t m_init = bz*block_size+tz;
    //                         size_t q_stride = block_size * grid_size;
    //                         size_t p_stride = block_size * grid_size;
    //                         size_t m_stride = block_size * grid_size;
                        
                        
    //                         //Begin convolution
    //                         for (size_t n = 0; n < OUTPUT_BATCHES; n++)
    //                             for (size_t q = q_init; q < OUTPUT_HEIGHT; q+=q_stride)
    //                                 for (size_t p = p_init; p < OUTPUT_WIDTH; p+=p_stride){
    //                                     for (size_t s = 0; s < LAYER_HEIGHT; s++)
    //                                         for (size_t r = 0; r < LAYER_WIDTH; r++)
    //                                             for (size_t c = 0; c < INPUT_CHANNELS; c++)
    //                                                 for (size_t m = m_init; m < OUTPUT_CHANNELS; m+=m_stride){
    //                                                     output.data[n][q][p][m] += input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c] * layer.weights[s][r][c][m];
    //                                                 }
    //                                         for (size_t m = m_init; m < OUTPUT_CHANNELS; m+=m_stride){
    //                                             output.data[n][q][p][m] += layer.bias[m];
    //                                             if(output.data[n][q][p][m] < 0) output.data[n][q][p][m] = 0;
    //                                         }
    //                                     }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

void conv4d_convolve_cuda_discrete_rewrite_gpu_data(int block_size, int grid_size){
    cudaCheckError();
    cuda_var_update();
    cudaCheckError();
    conv4d_convolve_cuda_discrete(block_size, grid_size);
    cudaCheckError();
}
