#pragma once

#include "conv4D.h"
#include "featuremap.h"

typedef struct conv_ret_t {
	double time_elapsed;
} conv_ret;

conv_ret conv4d_convolve_serial_naive(conv4d_layer layer, featuremap_3d input, featuremap_3d output);
conv_ret conv4d_convolve_serial_optimized(conv4d_layer layer, featuremap_3d input, featuremap_3d output, const size_t block_size);


#ifdef THREAD_SUPPORT
#include <pthread.h>
conv_ret conv4d_convolve_threads_naive(conv4d_layer layer, featuremap_3d input, featuremap_3d output);
conv_ret conv4d_convolve_threads_optimized(conv4d_layer layer, featuremap_3d input, featuremap_3d output, const size_t block_size);
#endif


#ifdef CUDA_SUPPORT
conv_ret conv4d_convolve_cuda_discrete(conv4d_layer layer, featuremap_3d input, featuremap_3d output);
conv_ret conv4d_convolve_cuda_CUDNN(conv4d_layer layer, featuremap_3d input, featuremap_3d output);
#endif


#ifdef OMP_SUPPORT
conv_ret conv4d_convolve_OpenMP_naive(conv4d_layer layer, featuremap_3d input, featuremap_3d output);
conv_ret conv4d_convolve_OpenMP_optimized(conv4d_layer layer, featuremap_3d input, featuremap_3d output, const size_t block_size);
#endif