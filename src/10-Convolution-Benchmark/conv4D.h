#pragma once

#include "featuremap.h"


typedef struct conv4d_layer_t {
	size_t stride_size;
	size_t kernel_width, kernel_height, output_channels, input_channels;
	float* weights, *bias;
} conv4d_layer;

/**
 * @brief Create a 3D feature map with the given convolution layer
 * 
 * @param layer 
 * @param input 
 * @return featuremap_3d 
 */
featuremap_3d conv4d_create_output(conv4d_layer layer, featuremap_3d input);

void conv4d_load(conv4d_layer* layer, FILE* weights_file, FILE* biases_file);

/**
 * @brief Deprecated. Do not use because it's slower than directly editing the value at an address with conv4d_weight_addr_of
 * 
 * @param layer 
 * @param output_channel 
 * @param input_channel 
 * @param kernel_x 
 * @param kernel_y 
 * @return float 
 */
float conv4d_weight_value_of(conv4d_layer layer, size_t output_channel, size_t input_channel, size_t kernel_x, size_t kernel_y);

float* conv4d_weight_addr_of(conv4d_layer layer, size_t output_channel, size_t input_channel, size_t kernel_x, size_t kernel_y);

float* conv4d_bias_addr_of(conv4d_layer layer, size_t output_channel);