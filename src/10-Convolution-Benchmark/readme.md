![Current screenshot](../../docs/screenshots/07-01.png)

# Project 7: Hello CUDA <!-- omit in toc -->


## Table of Contents <!-- omit in toc -->
- [CUDA C++](#cuda-c)
- [Walkthrough](#walkthrough)
- [Resources](#resources)
  - [Installing CUDA Tools](#installing-cuda-tools)
  - [Writing Applications in CUDA C++](#writing-applications-in-cuda-c)


## CUDA C++
CUDA C++ is a language that is *very* similar to C++. When compiled by `nvcc`, CUDA can be linked with standard C++ files to create GPU-accelerated programs.

CUDA implementation code usually has the file extension `.cu`, which tells CMake that it should use `nvcc` to compile the code. CUDA headers use the standard `.h` files used by C++ and C implementation code, so C++ is easily linkable with CUDA C++.


## Walkthrough
The heart of this simple CUDA application is a function that adds two numbers in an array.

This array can be very large (~1M elements in this program). Such a large number of elements would take a significant amount of time on a single-core CPU, but CUDA allows the operation to be broken into hundreds of smaller threads.

Here's how it's done:
```C++
// function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{  
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
```
* `__global__` tells the compiler that this function is intended to be run on a GPU and called from the CPU. `__global__` functions are known as *kernels*. *host code* (code that runs on the CPU) calls *kernels* that run *device code* (code that runs on the GPU)
* The CUDA compiler provides its own variables to `__global__` functions. Three of these are:
  *  `threadIdx.x` (the thread ID)
  *  `blockIdx.x` (the block ID of  that thread).
  *  `blockDim.x` (the number of blocks )
  *  `gridDim.x` (the number of threads in the grid)
*  The structure of this *for* loop is so common that it has a name: [the grid-stride loop](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/). 


## Resources
### Installing CUDA Tools
* [Official Nvidia instructions](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
### Writing Applications in CUDA C++ 
* [Build Systems: Combining CUDA and Modern CMake](https://on-demand.gputechconf.com/gtc/2017/presentation/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf)
* [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
* [Modern CMake: CUDA](https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html)
* [GPU Accelerated Computing with C and C++](https://developer.nvidia.com/how-to-cuda-c-cpp)