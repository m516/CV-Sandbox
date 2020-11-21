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

#define BENCHMARK_ALGORITHM(algo_name, ...) {                   \
	double time_elapsed = 0;                                    \
	double max_error = 0, e = 0;                                \
	unsigned long time_start, time_diff;                        \
	algo_name ( __VA_ARGS__ );                                  \
	max_error = (double)conv4d_total_error();                   \
	for (int i = 0; i < TRIALS; i++) {                          \
    	time_start = get_gtod_clock_time ();                    \
		algo_name ( __VA_ARGS__ );                              \
		time_diff = get_gtod_clock_time () - time_start;        \
		time_elapsed += (double) time_diff;                     \
	}                                                           \
	time_elapsed /= 1000000 * TRIALS;                           \
	printf(#algo_name ",\t%lf,\t%lf,\t" #__VA_ARGS__ "\n",      \
		time_elapsed, max_error);                               \
}                                                               \


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

	//BENCHMARK_ALGORITHM(conv4d_convolve_serial_naive);
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_discrete);
	//for(int i = 1; i < 10; i++)
	//	BENCHMARK_ALGORITHM(conv4d_convolve_serial_tiled, 1);
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_discrete);
}


