#include "conv4D_data_structures.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef USE_FILES
#if HAVE_UNISTD_H
#   include <unistd.h>
#elif _WIN32
#	include <direct.h>
#   define chdir _chdir
#else
#	warning "Couldn't find unistd.h via CMake and not a Windows platform. Attempting to use unistd.h anyway"
#   include <unistd.h>
#endif
#else
#   include "conv4D_impl.h"
#   include <string.h>
#endif

input_feature_map_t input;
output_feature_map_t output;
output_feature_map_t output_expected;
conv4d_layer_t layer;

const float* flattened_input = &input.data[0][0][0][0];
const float* flattened_output = &output.data[0][0][0][0];
const float* flattened_output_expected = &output_expected.data[0][0][0][0];




/**
 * Opens a file and returns the FILE pointer.
 * Make sure to fclose() the file.
 * 
 * \param filename the relative path of the file
 * \return 
 */
void write_file_to_float_array(const char* filename, float* destination, size_t elements) {
    //Sanity checking
    if(filename==NULL) CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("Filename is null\n");
    if(destination==NULL) CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("destination is null\n");

    //Load the file
	FILE* f;
	#if _MSC_VER && !__INTEL_COMPILER
	errno_t err = fopen_s(&f, filename, "rb");
	if (err) {
		CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("Failed to load file: %s\n", filename);
	}
	#else
	f = fopen(filename, "rb");
	if (f==NULL) {
		CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("Failed to load file: %s\n", filename);
	}
	#endif

    //Dump the contents in the destination
    size_t s = fread(destination, sizeof(float), elements, f);
    if(s!=elements){
        CONV4D_DATA_STRUCTURE_RUNTIME_ERROR("When loading from the file %s,\nFailed to load all %zu elements into the destination. Only got %zu\n", filename, elements, s);
    }
}

/**
 * @brief Overwrites the existing data in a floating point array with random numbers from 0 to a mximum value.
 * 
 * @param dest 
 * @param max_value 
 */
void randomize_float_array(float* dest, size_t amount, float max_value){
    for(size_t i = 0; i < amount; i++){
        *(dest++) = (float)((double)rand()/(double)((float)RAND_MAX/max_value));
    }
}


void conv4d_data_load(){
    #ifdef USE_FILES
	chdir(MEDIA_DIRECTORY);
    write_file_to_float_array(INPUT_FILENAME,        &input.data[0][0][0][0],           INPUT_SIZE);
    write_file_to_float_array(OUTPUT_FILENAME,       &output_expected.data[0][0][0][0], OUTPUT_SIZE);
    write_file_to_float_array(LAYER_WEIGHT_FILENAME, &layer.weights[0][0][0][0],        LAYER_WEIGHT_SIZE);
    write_file_to_float_array(LAYER_BIAS_FILENAME,   &layer.bias[0],                    LAYER_BIAS_SIZE);
    #else
    //Generate values for the input and layers.
    randomize_float_array(&input.data[0][0][0][0],    INPUT_SIZE,        1);
    randomize_float_array(&layer.weights[0][0][0][0], LAYER_WEIGHT_SIZE, 1);
    randomize_float_array(&layer.bias[0],             LAYER_BIAS_SIZE,   1);
    //Generate the output using the serial discrete version (assumed to be correct)
    conv4d_convolve_serial_discrete();
    //Copy the data over to output_expected
    memcpy(&output_expected, &output, sizeof(output_feature_map_t));
    #endif
}



long double conv4d_average_error(){
    //Total error
    long double total_error = 0, max_error = .0000001;

    //Get difference between feature maps
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = 0; q < OUTPUT_HEIGHT; q++)
            for (size_t p = 0; p < OUTPUT_WIDTH; p++)
                for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
		            long double delta = (long double)output.data[n][q][p][m] - output_expected.data[n][q][p][m];
                    if (delta < 0) delta = -delta;
                    //Increment error
                    total_error += delta;
                    //Get max error
                    if(delta>=max_error){
                        max_error=delta;
                        //printf("E: output[%zu][%zu][%zu][%zu] (%Lf)\n", n, q, p, m, max_error);
                    }
                }
    //Return sum of error
	return total_error / ((float)OUTPUT_SIZE);
}