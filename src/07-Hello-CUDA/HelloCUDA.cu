#include <stdio.h>

#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
      printf("Cuda error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  }

struct Test {
    char array[5];
};

__device__ Test dev_test; //dev_test is now global, statically allocated, and one instance of the struct

__global__ void kernel() {
    for(int i=0; i < 5; i++) {
        printf("Kernel[0][i]: %c \n", dev_test.array[i]);
    }
}


int main(void) {

    int size = 5;
    Test test; //test is now statically allocated and one instance of the struct

    char temp[] = { 'a', 'b', 'c', 'd' , 'e' };
    memcpy(test.array, temp, size * sizeof(char));

    cudaCheckError();
    cudaMemcpyToSymbol(dev_test, &test, sizeof(Test));
    cudaCheckError();
    kernel<<<1, 1>>>();
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    //  memory free
    return 0;
}