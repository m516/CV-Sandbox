#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "conv4D.h"
#include "conv4D_impl.h"
#include "featuremap.h"

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

#define BENCHMARK_ALGORITHM(algo_name, ...) { \
	double t = 0; \
	conv_ret r; \
	for (int i = -1; i < TRIALS; i++) { \
		r = algo_name ( __VA_ARGS__ ); \
		if (i < 0) continue; \
		t += r.time_elapsed; \
	} \
	t /= TRIALS; \
	printf("%lf", t); \
}                                                   \


/**
 * Opens a file and returns the FILE pointer.
 * Make sure to fclose() the file.
 * 
 * \param filename the relative path of the file
 * \return 
 */
FILE* open_file(const char* filename) {
	FILE* f;

	#if _MSC_VER && !__INTEL_COMPILER
	errno_t err = fopen_s(&f, filename, "rb");
	if (err) {
		printf_s("\nFailed to read file. Error code: %d\n", err);
		exit(1);
	}
	#else
	f = fopen(filename, "rb");
	if (f==NULL) {
		printf("\nFailed to read file");
		exit(1);
	}
	#endif


	return f;
}


void print_barrier() {
	printf("\n-------------------------------------------------\n\n");
}

void print_validation(featuremap_3d fmap, const char* comparison_filename) {
	//Print extra new line in case it hasn't been printed yet.
	printf("\n");

	//Activation
	featuremap_3d_activate_ReLU(fmap);

	//Print off first 25 elements of feature map
	//for (int i = 0; i < 25; i++) {
	//	printf("Element %7d: %1.3f\n", i, output.data[i]);
	//}
	//for (int i = featuremap_3d_size(output) - 25; i <= featuremap_3d_size(output); i++) {
	//	printf("Element %7d: %1.3f\n", i, output.data[i]);
	//}

	//Compare
	//Load the feature map from a file
	FILE* test_file = open_file(comparison_filename);
	featuremap_3d_floating_error e = featuremap_3d_compare_with_file(fmap, test_file);
	fclose(test_file);
	//Print
	featuremap_3d_print_error_information(e);
}



int main() {
	chdir(MEDIA_DIRECTORY);

	//Create the input feature map
	featuremap_3d input = {
		.batches = 1,
		.channels = 32,
		.width = 60,
		.height = 60,
		.data = 0
	};

	//Load the feature map from a file
	FILE* input_file = open_file("dnn/Test_Input0/layer_0_output.bin");
	featuremap_3d_load_from_file(&input, input_file);
	fclose(input_file);

	//Create the convolutional layer
	conv4d_layer layer = {
		.input_channels = input.channels,
		.output_channels = 32,
		.stride_size = 1,
		.kernel_width = 5,
		.kernel_height = 5
	};

	//Load the weights and biases from files
	FILE* weight_file = open_file("dnn/Test_Input0/conv2_weights.bin");
	FILE* bias_file = open_file("dnn/Test_Input0/conv2_biases.bin");
	conv4d_load(&layer, weight_file, bias_file);
	fclose(weight_file);
	fclose(bias_file);

	//Generate output feature map
	featuremap_3d output = conv4d_create_output(layer, input);

	
	//Benchmarking
	//Naive Serial Benchmarking
	printf("\nSerial Simple Algorithm: ");
	BENCHMARK_ALGORITHM(conv4d_convolve_serial_naive, layer, input, output);
	print_validation(output, "dnn/Test_Input0/layer_1_output.bin");
	print_barrier();

	/*
	//Optimized Serial Benchmarking
	printf("\nSerial Tiled Algorithm\nBlock Size\tAvg. Time");
	for(int block_size = 1; block_size < 65; block_size ++){
		printf("\n\t%d\t", block_size);
		BENCHMARK_ALGORITHM(conv4d_convolve_serial_optimized, layer, input, output, block_size);
	}
	print_validation(output, "dnn/Test_Input0/layer_1_output.bin");
	print_barrier();
	*/

	//Threads
	#ifdef THREAD_SUPPORT
	//Naive OpenMP Benchmarking
	printf("\PThread Simple Algorithm ");
	BENCHMARK_ALGORITHM(conv4d_convolve_threads_naive, layer, input, output);
	print_validation(output, "dnn/Test_Input0/layer_1_output.bin");
	print_barrier();
	#endif

	//OpenMP
	#ifdef OMP_SUPPORT
	printf("\nNumber of processors: %d", omp_get_num_procs());
	omp_set_num_threads(omp_get_num_procs());

	//Naive OpenMP Benchmarking
	printf("\nOpenMP Simple Algorithm ");
	BENCHMARK_ALGORITHM(conv4d_convolve_OpenMP_naive, layer, input, output);
	print_validation(output, "dnn/Test_Input0/layer_1_output.bin");
	print_barrier();


	#endif


	return 0;
}


