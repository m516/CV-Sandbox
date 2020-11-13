#pragma once
#include <stdio.h>
#include <float.h>

typedef struct featuremap_3d_t {
	float* data;
	size_t batches;
	size_t channels, width, height;
} featuremap_3d;

typedef struct featuremap_3d_floating_error_t {
	long double max_error, average_error, min_epsilon;
	size_t index_of_max_error;
}featuremap_3d_floating_error;

size_t featuremap_3d_size(featuremap_3d featuremap);

void featuremap_3d_set_value_of(featuremap_3d featuremap, size_t batch, size_t channel, size_t x, size_t y, float new_value);

float featuremap_3d_value_of(featuremap_3d featuremap, size_t batch, size_t channel, size_t x, size_t y);

float* featuremap_3d_addr_of(featuremap_3d featuremap, size_t batch, size_t channel, size_t x, size_t y);

void featuremap_3d_load_from_file(featuremap_3d* featuremap, FILE* file);

featuremap_3d_floating_error featuremap_3d_compare_with_file(featuremap_3d featuremap, FILE* file);

featuremap_3d_floating_error featuremap_3d_compare(featuremap_3d featuremap1, featuremap_3d featuremap2);

void featuremap_3d_print_error_information(featuremap_3d_floating_error e);

void featuremap_3d_activate_ReLU(featuremap_3d featuremap);