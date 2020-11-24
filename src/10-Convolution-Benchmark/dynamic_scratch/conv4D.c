//Standard libraries
#include <stdlib.h>
#include <assert.h>

//Local headers
#include "conv4D.h"
#include "featuremap.h"



featuremap_3d conv4d_create_output(conv4d_layer layer, featuremap_3d input)
{
	featuremap_3d f;
	f.batches = input.batches;
	f.width = (input.width - layer.kernel_width + layer.stride_size) / layer.stride_size;
	f.height = (input.height - layer.kernel_height + layer.stride_size) / layer.stride_size;
	f.channels = layer.output_channels;
    f.data = malloc(featuremap_3d_size(f) * sizeof(float));

	return f;
}

void conv4d_load(conv4d_layer* layer, FILE* weights_file, FILE* biases_file)
{
    size_t weight_size = layer->output_channels * layer->input_channels * layer->kernel_width * layer->kernel_height;
    layer->weights = malloc(weight_size * sizeof(float));
    layer->bias = malloc(layer->output_channels * sizeof(float));
    assert(layer->weights != 0);
    assert(layer->bias != 0);
    size_t s;
    s = fread(layer->weights, sizeof(float), weight_size, weights_file);
    assert(s == weight_size);
	s = fread(layer->bias, sizeof(float), layer->output_channels, biases_file);
    assert(s == layer->output_channels);
}

float conv4d_weight_value_of(conv4d_layer layer, size_t output_channel, size_t input_channel, size_t kernel_x, size_t kernel_y)
{
    return layer.weights[
        kernel_y * layer.output_channels * layer.input_channels * layer.kernel_width
        + kernel_x * layer.output_channels * layer.input_channels
        + input_channel * layer.output_channels
        + output_channel
        ];
}

float* conv4d_weight_addr_of(conv4d_layer layer, size_t output_channel, size_t input_channel, size_t kernel_x, size_t kernel_y) {
    return layer.weights
        + kernel_y * layer.output_channels * layer.input_channels * layer.kernel_width
        + kernel_x * layer.output_channels * layer.input_channels
        + input_channel * layer.output_channels
        + output_channel;
}

float* conv4d_bias_addr_of(conv4d_layer layer, size_t output_channel) {
    return layer.bias + output_channel;
}