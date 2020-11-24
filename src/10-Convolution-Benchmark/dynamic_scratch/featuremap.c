#include "featuremap.h"
#include <stdlib.h>
#include <assert.h>


size_t featuremap_3d_size(featuremap_3d featuremap) {
	return featuremap.batches * featuremap.channels * featuremap.width * featuremap.height;
}

void featuremap_3d_set_value_of(featuremap_3d featuremap, size_t batch, size_t channel, size_t x, size_t y, float new_value)
{
	featuremap.data[
		batch * featuremap.channels * featuremap.width * featuremap.height
			+ y * featuremap.channels * featuremap.width
			+ x * featuremap.channels
			+ channel
	] = new_value;
	return;
}

float featuremap_3d_value_of(featuremap_3d featuremap, size_t batch, size_t channel, size_t x, size_t y)
{
	return featuremap.data[
		batch * featuremap.channels * featuremap.width * featuremap.height
		+ y * featuremap.channels * featuremap.width
		+ x * featuremap.channels
		+ channel
		];
}

float* featuremap_3d_addr_of(featuremap_3d featuremap, size_t batch, size_t channel, size_t x, size_t y) {
	return featuremap.data
		+ batch * featuremap.channels * featuremap.width * featuremap.height
		+ y * featuremap.channels * featuremap.width
		+ x * featuremap.channels
		+ channel;
}

void featuremap_3d_load_from_file(featuremap_3d* featuremap, FILE* file) {
	featuremap->data = malloc(featuremap_3d_size(*featuremap) * sizeof(float));
	assert(featuremap->data != 0);
	fread(featuremap->data, sizeof(float), featuremap_3d_size(*featuremap), file);
}

featuremap_3d_floating_error featuremap_3d_compare_with_file(featuremap_3d featuremap, FILE* file)
{
	featuremap_3d t = featuremap;
	featuremap_3d_load_from_file(&t, file);
	featuremap_3d_floating_error e = featuremap_3d_compare(featuremap, t);
	free(t.data);
	return e;
}

featuremap_3d_floating_error featuremap_3d_compare(featuremap_3d featuremap1, featuremap_3d featuremap2)
{
	featuremap_3d_floating_error e = {
		.max_error = 0,
		.average_error = 0,
		.min_epsilon = 0,
		.index_of_max_error = 0
	};

	assert(featuremap_3d_size(featuremap1) == featuremap_3d_size(featuremap2));

	for (size_t i = 0; i < featuremap_3d_size(featuremap1); i++) {
		//Get difference between feature maps
		long double delta = (long double)featuremap1.data[i] - featuremap2.data[i];
		if (delta < 0) delta = -delta;
		//Update components
		e.average_error += delta;
		if (e.max_error < delta) {
			e.max_error = delta;
			e.index_of_max_error = i;
		}
	}
	//Finishing touches
	printf("e.average_error: %Le\n", e.average_error);
	printf("featuremap_3d_size(featuremap1): %zu\n", featuremap_3d_size(featuremap1));
	if (featuremap_3d_size(featuremap1) == 0) {
		printf("WARNING: division by 0\n");
	}

	//Watch out, can't convert from unsigned int to floating point
	//numbers when AVX512 instruction set is enabled
	long double t = e.average_error / (long double)(long)featuremap_3d_size(featuremap1);
	e.average_error = t;
	e.min_epsilon = e.max_error / FLT_EPSILON;
	return e;
}

void featuremap_3d_print_error_information(featuremap_3d_floating_error e) {
	printf(" * Average error: %Le\n", e.average_error);
	printf(" *    Max. error: %Le\n", e.max_error);
	printf("     at location  %zu\n", e.index_of_max_error);
	printf(" *  Min. epsilon: %Le\n", e.min_epsilon);
}

void featuremap_3d_activate_ReLU(featuremap_3d featuremap)
{
	for (size_t i = 0; i < featuremap_3d_size(featuremap); i++) {
		if (featuremap.data[i] < 0) featuremap.data[i] = 0;
	}
}
