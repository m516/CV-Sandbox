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

#define TRIALS 10


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
		printf_s("Failed to read file. Error code: %d\n", err);
		exit(1);
	}
	#else
	f = fopen(filename, "rb");
	#endif


	return f;
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

	//Print off first 25 elements of feature map
	//for (int i = 0; i < 25; i++) {
	//	printf("Element %7d: %1.3f\n", i, layer.weights[i]);
	//}

	//Generate output feature map
	featuremap_3d output = conv4d_create_output(layer, input);

	//Benchmarking
	//Naive Benchmarking
	printf("Naive Benchmarking\n");
	double t = 0; //Total time
	conv_ret r;
	for(int i = -1; i < TRIALS; i ++){
		r = conv4d_convolve_serial_naive(layer, input, output);
		//Ignore first trial
		if(i<0) continue;
		t += r.time_elapsed;
	}
	t /= TRIALS;
	printf("%lf\n", t);

	//Optimized Benchmarking
	printf("Optimized Benchmarking\n");
	printf("Block Size\tAvg. Time\n");
	for(int block_size = 1; block_size < 65; block_size ++){
		t = 0; //Total time
		for(int i = -1; i < TRIALS; i ++){
			r = conv4d_convolve_serial_optimized(layer, input, output, block_size);
			//Ignore first trial
			if(i<0) continue;
			t += r.time_elapsed;
		}
		t /= TRIALS;
		printf("%d\t%lf\n", block_size, t);
	}

	

	//Activation
	featuremap_3d_activate_ReLU(output);

	//Print off first 25 elements of feature map
	//for (int i = 0; i < 25; i++) {
	//	printf("Element %7d: %1.3f\n", i, output.data[i]);
	//}
	//for (int i = featuremap_3d_size(output) - 25; i <= featuremap_3d_size(output); i++) {
	//	printf("Element %7d: %1.3f\n", i, output.data[i]);
	//}

	//Compare
	//Load the feature map from a file
	FILE* test_file = open_file("dnn/Test_Input0/layer_1_output.bin");
	featuremap_3d_floating_error e = featuremap_3d_compare_with_file(output, test_file);
	fclose(test_file);
	//Print
	featuremap_3d_print_error_information(e);

	return 0;
}


