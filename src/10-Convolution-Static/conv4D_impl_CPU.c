#include "conv4D_data_structures.h"
#include "conv4D_impl.h"
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

void conv4d_convolve_serial_naive(){
    //Reset memory
    memset(output.data, 0, OUTPUT_SIZE * sizeof(float));

    //Begin convolution
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = 0; q < OUTPUT_HEIGHT; q++)
            for (size_t p = 0; p < OUTPUT_WIDTH; p++)
                for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
                    for (size_t s = 0; s < LAYER_HEIGHT; s++)
                        for (size_t r = 0; r < LAYER_WIDTH; r++)
                            for (size_t c = 0; c < INPUT_CHANNELS; c++)
                                output.data[n][q][p][m] += input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c] * layer.weights[s][r][c][m];
                output.data[n][q][p][m] += layer.bias[m];
                if(output.data[n][q][p][m] < 0) output.data[n][q][p][m] = 0;
                }

}
void conv4d_convolve_serial_discrete(){
    //Reset memory
    memset(output.data, 0, OUTPUT_SIZE * sizeof(float));

    //Begin convolution
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = 0; q < OUTPUT_HEIGHT; q++)
            for (size_t p = 0; p < OUTPUT_WIDTH; p++)
                for (size_t s = 0; s < LAYER_HEIGHT; s++)
                    for (size_t r = 0; r < LAYER_WIDTH; r++)
                        for (size_t c = 0; c < INPUT_CHANNELS; c++)
                            for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
                                output.data[n][q][p][m] += input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c] * layer.weights[s][r][c][m];
                                //printf("%zu\t%zu\t%zu\n", 
                                //&output.data[n][q][p][m]-flattened_output,
                                //&input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c]-flattened_input,
                                //&layer.weights[s][r][c][m]-&layer.weights[0][0][0][0]);
                                //usleep(10000);
                            }

    //Bias and activation function (ReLU)
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = 0; q < OUTPUT_HEIGHT; q++)
            for (size_t p = 0; p < OUTPUT_WIDTH; p++)
                for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
                    output.data[n][q][p][m] += layer.bias[m];
                    if(output.data[n][q][p][m] < 0) output.data[n][q][p][m] = 0;
                }
}

void conv4d_convolve_serial_tiled(int block_size){
    //Reset memory
    memset(output.data, 0, OUTPUT_SIZE * sizeof(float));

    //Begin convolution
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q0 = 0; q0 < OUTPUT_HEIGHT; q0+=block_size)
            for (size_t p0 = 0; p0 < OUTPUT_WIDTH; p0+=block_size)
                for (size_t s = 0; s < LAYER_HEIGHT; s++)
                    for (size_t r = 0; r < LAYER_WIDTH; r++)
                        for (size_t m = 0; m < OUTPUT_CHANNELS; m++)
                            for (size_t c = 0; c < INPUT_CHANNELS; c++)
                                for(size_t q1 = 0; q1 < block_size; q1++){
                                    size_t q=q0+q1;
                                    if(q>OUTPUT_HEIGHT) break;
                                    for(size_t p1 = 0; p1 < block_size; p1++){
                                        size_t p=p0+p1;
                                        if(p>OUTPUT_WIDTH)  break;
                                        output.data[n][q][p][m] += input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c] * layer.weights[s][r][c][m];
                                        //printf("%zu\t%zu\t%zu\n", 
                                        //&output.data[n][q][p][m]-flattened_output,
                                        //&input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c]-flattened_input,
                                        //&layer.weights[s][r][c][m]-&layer.weights[0][0][0][0]);
                                        //usleep(10000);
                                    }
                                }
                                    
                                        

    //Bias and activation function (ReLU)
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = 0; q < OUTPUT_HEIGHT; q++)
            for (size_t p = 0; p < OUTPUT_WIDTH; p++)
                for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
                    output.data[n][q][p][m] += layer.bias[m];
                    if(output.data[n][q][p][m] < 0) output.data[n][q][p][m] = 0;
                }
}

#ifdef THREAD_SUPPORT
void* conv4d_convolve_threads_discrete_helper(){
    static int thread_index = 0;
    
    //Get the thread ID
    static pthread_mutex_t* mutex = NULL;
    if(mutex==NULL){
        mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
        pthread_mutex_init(mutex, NULL);
    }
    pthread_mutex_lock(mutex);
    int my_thread_index = thread_index++;
    pthread_mutex_unlock(mutex);

    //Begin convolution
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = my_thread_index; q < OUTPUT_HEIGHT; q+=THREAD_SUPPORT)
            for (size_t p = 0; p < OUTPUT_WIDTH; p++)
                for (size_t s = 0; s < LAYER_HEIGHT; s++)
                    for (size_t r = 0; r < LAYER_WIDTH; r++)
                        for (size_t c = 0; c < INPUT_CHANNELS; c++)
                            for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
                                output.data[n][q][p][m] += input.data[n][q*LAYER_STRIDE+s][p*LAYER_STRIDE+r][c] * layer.weights[s][r][c][m];
                            }
    return NULL;

}

void conv4d_convolve_threads_discrete(){
    //Reset memory
    memset(output.data, 0, OUTPUT_SIZE * sizeof(float));

    pthread_t threads[THREAD_SUPPORT];

    for(int i = 0; i < THREAD_SUPPORT; i++){
        pthread_create(threads+i, NULL, conv4d_convolve_threads_discrete_helper, NULL);
    }
    for(int i = 0; i < THREAD_SUPPORT; i++){
        pthread_join(threads[i], NULL);
    }

    //Bias and activation function (ReLU)
    for (size_t n = 0; n < OUTPUT_BATCHES; n++)
        for (size_t q = 0; q < OUTPUT_HEIGHT; q++)
            for (size_t p = 0; p < OUTPUT_WIDTH; p++)
                for (size_t m = 0; m < OUTPUT_CHANNELS; m++){
                    output.data[n][q][p][m] += layer.bias[m];
                    if(output.data[n][q][p][m] < 0) output.data[n][q][p][m] = 0;
                }
}
void conv4d_convolve_threads_tiled(int block_size){
    //TODO
}
#endif

#ifdef OMP_SUPPORT
void conv4d_convolve_OpenMP_discrete(){
    //TODO
}
void conv4d_convolve_OpenMP_tiled(int block_size){
    //TODO
}
#endif