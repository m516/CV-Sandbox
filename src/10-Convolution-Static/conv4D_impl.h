#pragma once

#include "conv4D_data_structures.h"

void conv4d_convolve_serial_naive();
void conv4d_convolve_serial_discrete();
void conv4d_convolve_serial_tiled(int block_size);

#ifdef THREAD_SUPPORT
#include <pthread.h>
void conv4d_convolve_threads_discrete();
void conv4d_convolve_threads_tiled(int block_size);
#endif


#ifdef CUDA_SUPPORT
void conv4d_convolve_cuda_discrete();
void conv4d_convolve_cuda_CUDNN();
#endif


#ifdef OMP_SUPPORT
#include <omp.h>
void conv4d_convolve_OpenMP_discrete();
void conv4d_convolve_OpenMP_tiled(int block_size);
#endif