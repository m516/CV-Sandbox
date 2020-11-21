#pragma once

#define CONV4D_DATA_STRUCTURE_RUNTIME_ERROR(...){                     \
fprintf(stderr, "Runtime error regarding a Conv4D data structure\n"); \
fprintf(stderr, "Encountered on %s:%d\n", __FILE__, __LINE__);        \
fprintf(stderr, "In function %s\n", __func__);                        \
fprintf(stderr, __VA_ARGS__);                                         \
exit(1);                                                              \
}                                                                     
//Filenames
#define INPUT_FILENAME        "dnn/Test_Input0/layer_0_output.bin"
#define OUTPUT_FILENAME       "dnn/Test_Input0/layer_1_output.bin"
#define LAYER_WEIGHT_FILENAME "dnn/Test_Input0/conv2_weights.bin"
#define LAYER_BIAS_FILENAME   "dnn/Test_Input0/conv2_biases.bin"
//Input parameters
#define INPUT_BATCHES 1    //Configure for each layer
#define INPUT_WIDTH 60     //Configure for each layer
#define INPUT_HEIGHT 60    //Configure for each layer
#define INPUT_CHANNELS 32  //Configure for each layer
//Layer parameter
#define LAYER_WIDTH 5     //Configure for each layer
#define LAYER_HEIGHT 5    //Configure for each layer
#define LAYER_STRIDE 1    //Configure for each layer
//Output parameter
#define OUTPUT_CHANNELS 32 //Configure for each layer

//Calculated output dimensions
#define OUTPUT_BATCHES INPUT_BATCHES
#define OUTPUT_WIDTH  (INPUT_WIDTH - LAYER_WIDTH  + LAYER_STRIDE) / LAYER_STRIDE
#define OUTPUT_HEIGHT (INPUT_HEIGHT- LAYER_HEIGHT + LAYER_STRIDE) / LAYER_STRIDE
//Sizing
#define INPUT_SIZE INPUT_BATCHES * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS
#define OUTPUT_SIZE OUTPUT_BATCHES * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS
#define LAYER_WEIGHT_SIZE LAYER_HEIGHT * LAYER_WIDTH * INPUT_CHANNELS * OUTPUT_CHANNELS
#define LAYER_BIAS_SIZE OUTPUT_CHANNELS

typedef struct input_feature_map{
    float data[INPUT_BATCHES][INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS];
} input_feature_map_t;

typedef struct output_feature_map{
    float data[OUTPUT_BATCHES][OUTPUT_HEIGHT][OUTPUT_WIDTH][OUTPUT_CHANNELS];
} output_feature_map_t;

typedef struct conv4d_layer{
    float weights[LAYER_HEIGHT][LAYER_WIDTH][INPUT_CHANNELS][OUTPUT_CHANNELS];
    float bias[OUTPUT_CHANNELS];
} conv4d_layer_t;

extern input_feature_map_t input;
extern output_feature_map_t output;
extern conv4d_layer_t layer;
extern const float* flattened_input;
extern const float* flattened_output;
extern const float* flattened_output_expected;

void conv4d_data_load();
long double conv4d_total_error();
