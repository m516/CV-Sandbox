#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include "conv4D_data_structures.h"
#include "conv4D_impl.h"

#if HAVE_UNISTD_H
#   include <unistd.h>
#elif _WIN32
#	include <direct.h>
#   define chdir _chdir
#else
#	warning "Couldn't find unistd.h via CMake and not a Windows platoform. Attempting to use unistd.h anyway"
#   include <unistd.h>
#endif


#ifdef OMP_SUPPORT
#include <omp.h>
#endif
#ifdef THREAD_SUPPORT
#include <pthread.h>
#endif

#define TRIALS 10

#define BENCHMARK_ALGORITHM(algo_name, ...) {                                                         \
	double benchmark_algorithm_time_elapsed = 0;                                                     \
	double benchmark_algorithm_max_error = 0;                                                        \
	unsigned long benchmark_algorithm_time_start;                                                    \
	unsigned long benchmark_algorithm_time_diff;                                                     \
	algo_name ( __VA_ARGS__ );                                                                        \
	benchmark_algorithm_max_error = (double)conv4d_total_error();                                    \
	for (int benchmark_algorithm_i = 0;                                                              \
			benchmark_algorithm_i < TRIALS;                                                          \
			benchmark_algorithm_i++) {                                                               \
    	benchmark_algorithm_time_start = get_gtod_clock_time ();                                     \
		algo_name ( __VA_ARGS__ );                                                                    \
		benchmark_algorithm_time_diff = get_gtod_clock_time () - benchmark_algorithm_time_start;    \
		benchmark_algorithm_time_elapsed += (double) benchmark_algorithm_time_diff;                 \
	}                                                                                                 \
	benchmark_algorithm_time_elapsed /= 1000000 * TRIALS;                                            \
	printf(#algo_name ",\t%lf,\t%lf,\t" #__VA_ARGS__ "\n",                                            \
		benchmark_algorithm_time_elapsed, benchmark_algorithm_max_error);                           \
}                                                                                                     \


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
	chdir(MEDIA_DIRECTORY);

	conv4d_data_load();

	int i;

	
	/*
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
	*/



	#ifdef CUDA_SUPPORT
	BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 4);
	BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 5);
	BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 6);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 7);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 8);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 9);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 10);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 11);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 12);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 13);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 14);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 15);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 16);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 17);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 18);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 19);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 20);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 21);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 22);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 23);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 24);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 25);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 26);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 27);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 28);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 29);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 30);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 31);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete_rewrite_gpu_data, 32);
	BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 4);
	BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 5);
	BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 6);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 7);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 8);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 9);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 10);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 11);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 12);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 13);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 14);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 15);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 16);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 17);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 18);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 19);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 20);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 21);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 22);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 23);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 24);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 25);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 26);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 27);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 28);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 29);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 30);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 31);
	// BENCHMARK_ALGORITHM(conv4d_convolve_cuda_discrete, 32);
	#endif

}


