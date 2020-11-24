#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "conv4D_data_structures.h"
#include "conv4D_impl.h"



#ifdef OMP_SUPPORT
#include <omp.h>
#endif
#ifdef THREAD_SUPPORT
#include <pthread.h>
#endif

#define TRIALS 32

#define BENCHMARK_ALGORITHM(algo_name, ...) {                                                        \
	double benchmark_algorithm_time_elapsed = 0;                                                     \
	double benchmark_algorithm_max_error = 0;                                                        \
	unsigned long benchmark_algorithm_time_start;                                                    \
	unsigned long benchmark_algorithm_time_diff;                                                     \
	algo_name ( __VA_ARGS__ );                                                                       \
	benchmark_algorithm_max_error = (double)conv4d_average_error();                                  \
	for (int benchmark_algorithm_i = 0;                                                              \
			benchmark_algorithm_i < TRIALS;                                                          \
			benchmark_algorithm_i++) {                                                               \
    	benchmark_algorithm_time_start = get_gtod_clock_time ();                                     \
		algo_name ( __VA_ARGS__ );                                                                   \
		benchmark_algorithm_time_diff = get_gtod_clock_time () - benchmark_algorithm_time_start;     \
		benchmark_algorithm_time_elapsed += (double) benchmark_algorithm_time_diff;                  \
	}                                                                                                \
	benchmark_algorithm_time_elapsed /= 1000000 * TRIALS;                                            \
	printf(#algo_name ",\t%lf,\t%lf,\t" #__VA_ARGS__ "\n",                                           \
		benchmark_algorithm_time_elapsed, benchmark_algorithm_max_error);                            \
}                                                                                                    \


unsigned long get_gtod_clock_time ()
{
    struct timeval tv;

    if (gettimeofday (&tv, NULL) == 0)
        return (unsigned long) (tv.tv_sec * 1000000 + tv.tv_usec);
    else
        return 0;
}


void print_barrier() {
	printf("\n-------------------------------------------------\n\n");
}



int main() {

	printf("Floating point operations: %d\n", (OUTPUT_BATCHES)*(OUTPUT_HEIGHT)*((OUTPUT_WIDTH)*LAYER_HEIGHT*LAYER_WIDTH*INPUT_CHANNELS+1)*OUTPUT_CHANNELS);

	conv4d_data_load();
	
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_naive);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_discrete);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 1);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 2);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 3);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 4);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 5);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 6);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 7);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 8);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 9);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 10);

	#ifdef THREAD_SUPPORT
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_discrete);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 1);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 2);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 3);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 4);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 5);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 6);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 7);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 8);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 9);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_tiled, 10);
	#endif
	
	#ifdef OMP_SUPPORT
	omp_set_num_threads(THREAD_SUPPORT);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_discrete);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 1);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 2);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 3);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 4);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 5);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 6);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 7);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 8);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 9);
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_tiled, 10);
	#endif

	 #ifdef CUDA_SUPPORT
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete_rewrite_gpu_data, 4, 4);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 1,   1);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 2,   1);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 4,   1);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 8,   1);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 1,  2);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 2,  2);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 4,  2);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 8,  2);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 1,  4);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 2,  4);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 4,  4);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 8,  4);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 1,  8);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 2,  8);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 4,  8);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 8,  8);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 1,  16);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 2,  16);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 4,  16);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 8,  16);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 1,  32);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 2,  32);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 4,  32);
	BENCHMARK_ALGORITHM(conv4d_convolve_CUDA_discrete, 8,  32);
	#endif
}


