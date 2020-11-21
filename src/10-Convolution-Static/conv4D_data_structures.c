#include "conv4D_data_structures.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void conv4d_data_load(){
    write_file_to_float_array(INPUT_FILENAME,        &input.data[0][0][0][0],           INPUT_SIZE);
    write_file_to_float_array(OUTPUT_FILENAME,       &output_expected.data[0][0][0][0], OUTPUT_SIZE);
    write_file_to_float_array(LAYER_WEIGHT_FILENAME, &layer.weights[0][0][0][0],        LAYER_WEIGHT_SIZE);
    write_file_to_float_array(LAYER_BIAS_FILENAME,   &layer.bias[0],                    LAYER_BIAS_SIZE);
}

long double conv4d_total_error(){
    //Total error
    long double total_error = 0;

    //For each element in the output array
    for (size_t i = 0; i < OUTPUT_SIZE; i++) {
		//Get difference between feature maps
		long double delta = (long double)flattened_output[i] - flattened_output_expected[i];
		if (delta < 0) delta = -delta;
		//Increment error
        total_error += delta;
	}

    //Return sum of error
	return total_error;
}