#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define cudaCheckError() { \
  cudaError_t err = cudaGetLastError(); \
  if(err != cudaSuccess) { \
    printf("Cuda error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
}
