#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "conv4D_impl.h"

conv_ret conv4d_convolve_serial_naive(conv4d_layer layer, featuremap_3d input, featuremap_3d output)
{
    //Reset memory
    memset(output.data, 0, featuremap_3d_size(output) * sizeof(float));
    
    //Benchmarking setup
    conv_ret r;
    clock_t start_t, end_t;
    start_t = clock();

    //Begin convolution
    for (size_t n = 0; n < output.batches; n++)
        for (size_t q = 0; q < output.height; q++)
            for (size_t p = 0; p < output.width; p++)
                for (size_t s = 0; s < layer.kernel_height; s++)
                    for (size_t r = 0; r < layer.kernel_width; r++)
                        for (size_t c = 0; c < input.channels; c++)
                            for (size_t m = 0; m < output.channels; m++)
                            {
                                size_t i_index = n * input.channels * input.width * input.height
                                                + (layer.stride_size * q + s) * input.channels * input.width
                                                + (layer.stride_size * p + r) * input.channels
                                                + c;
                                size_t o_index = n * output.channels * output.width * output.height
                                                + q * output.channels * output.width
                                                + p * output.channels
                                                + m;
                                size_t f_index = s * layer.output_channels * layer.input_channels * layer.kernel_width
                                                + r * layer.output_channels * layer.input_channels
                                                + c * layer.output_channels
                                                + m;
                                float i = input.data[i_index];
                                float f = layer.weights[f_index];
                                output.data[o_index] += i * f;
                                //printf("%zu %zu %zu\n", i_index, f_index, o_index);
                            }

    //Bias
    for (size_t n = 0; n < output.batches; n++)
        for (size_t q = 0; q < output.height; q++)
            for (size_t p = 0; p < output.width; p++)
                for (size_t m = 0; m < output.channels; m++)
                    *(featuremap_3d_addr_of(output, n, m, p, q)) += layer.bias[m];
    
    //End benchmarking         
    end_t = clock();
    r.time_elapsed = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    return r;
}

conv_ret conv4d_convolve_serial_optimized(conv4d_layer layer, featuremap_3d input, featuremap_3d output, const size_t block_size)
{

    //Reset memory
    memset(output.data, 0, featuremap_3d_size(output) * sizeof(float));

    //Benchmarking setup
    conv_ret r;
    clock_t start_t, end_t;
    start_t = clock();

    unsigned int iteration = 0;

    //Convolve
    size_t n, s, q;
    for (size_t n0 = 0; n0 < output.batches; n0 += block_size)
        for (size_t q0 = 0; q0 < output.height; q0 += block_size)
            for (size_t s0 = 0; s0 < layer.kernel_height; s0+=block_size)
                for (size_t p = 0; p < output.width; p++)
                    for (size_t r = 0; r < layer.kernel_width; r++)
                        for (size_t c = 0; c < input.channels; c++)
                            for (size_t m = 0; m < output.channels; m++)
                                //Blocking over n, q, and p
                                for (size_t n1 = 0; n1 < block_size && (n=n0+n1) < output.batches; n1++)
                                    for (size_t q1 = 0; q1 < block_size && (q=q0+q1) < output.height; q1++)
                                        for (size_t s1 = 0; s1 < block_size && (s=s0+s1) < layer.kernel_height; s1++){
                                            size_t i_index = n * input.channels * input.width * input.height
                                                            + (layer.stride_size * q + s) * input.channels * input.width
                                                            + (layer.stride_size * p + r) * input.channels
                                                            + c;
                                            size_t o_index = n * output.channels * output.width * output.height
                                                            + q * output.channels * output.width
                                                            + p * output.channels
                                                            + m;
                                            size_t f_index = s * layer.output_channels * layer.input_channels * layer.kernel_width
                                                            + r * layer.output_channels * layer.input_channels
                                                            + c * layer.output_channels
                                                            + m;
                                            float i = input.data[i_index];
                                            float f = layer.weights[f_index];
                                            output.data[o_index] += i * f;
                                            //printf("%I32d %zu %zu %zu\n", iteration++, i_index, f_index, o_index);
                                        }
                                            

    //Bias
    for (size_t n = 0; n < output.batches; n++)
        for (size_t q = 0; q < output.height; q++)
            for (size_t p = 0; p < output.width; p++)
                for (size_t m = 0; m < output.channels; m++)
                    *(featuremap_3d_addr_of(output, n, m, p, q)) += layer.bias[m];


    //End benchmarking         
    end_t = clock();
    r.time_elapsed = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    return r;

}

#ifdef THREAD_SUPPORT
conv_ret conv4d_convolve_threads_naive(conv4d_layer layer, featuremap_3d input, featuremap_3d output)
{
    //Benchmarking setup
    conv_ret r;
    clock_t start_t, end_t;
    start_t = clock();
    //TODO stub
    //End benchmarking         
    end_t = clock();
    r.time_elapsed = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    return r;
}
conv_ret conv4d_convolve_threads_optimized(conv4d_layer layer, featuremap_3d input, featuremap_3d output, const size_t block_size)
{
    //Benchmarking setup
    conv_ret r;
    time_t start_t, end_t;
    time(&start_t);
    //TODO stub
    //End benchmarking         
    end_t = clock();
    r.time_elapsed = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    return r;
}
#endif

#ifdef OMP_SUPPORT

/*
OpenMP is a framework. Most issues stem from user
OpenMP is too easy. Sometimes ideas are easy and quick to implement. 
But some are more expensive than others.

If you write dumb code, you will get dumb performane. Don't just blame OpenMP

1. You must pay attention to single-thread performance.
It must perform reasonable well. If it doesn't, what will happen on 10 cores, 20 cores, ...?

Remember, scalability can mask poor performance.
A slow code tends to scale better, but is often still slower.

2. Do not parallelize what does NOT matter
Never tune your code without a profiling tool.
Blindly parallelizing code is never a good idea.
Q: What profiling tools do you use and why?
Don't share data unless you have to. Use private data as much as possible.
One "parallel for" is fine. Multiple back-to-back is EVIL
Think BIG and maximize the size of parallel regions.

What NOT to do
#pragma omp parallel for
{ <Code block 1> }
#pragma omp parallel for
{ <Code block 2> }
....
#pragma omp parallel for
{ <Code block n> }

Why?
Barriers are expensive: all threads wait for last one is finished (do nothing)


What to do (only one parallel region)
#pragma omp parallel
#pragma omp for
{ <Code block 1> }
#pragma omp for (nowait)
{ <Code block 2> }
....
#pragma omp for (nowait)
{ <Code block n> }
#pragma omp end parallel

Identify opportunities for nowait use.
(a powerful feature, but be aware of data races)
Use a schedule clause in case of load balancing issue
Use a profiling tool
Every barrier matters (same is true for locks and critical regions). 
Use atomic operations where possible.
At the end of the day, EVERYTHING matters


---------Memory Access---------------
The nice thing about OpenMP is that memory access "just happens"
However, there are two things to watch out for:
1. Non-Uniform Memory Access (NUMA)
2. False sharing
They have nothing to do with OpenMP and are a characteristic of using a shared memory architecture.
They are important and affect performance.

What is NUMA?
Memory is physically distributed, but logically shared.
Shared data is transparently accessible to all threads.
You don't know where the data is and shouldn't matter because the system finds it for you
It does matter:
    Each processor has its own memory. Processes run on single machines.
    NUMA systems allow processors to access other processor's memory.
    But there is an overhead to getting data from other processors.
As core and node count go up, but it increasingly matters.
This good news is that OpenMP has great support for NUMA

False Sharing
    Occurs when multiple threads modify the same block of data (cache line) at the same time
    Results in cache line moving through the system (an expensive operation)
    Additional cost of cache coherence updates
    Okay if it happens once in a while
    Very bad if frequent

    Example:
    #pragma omp parallel shared(a)
    {
        int TID = omp_get_thread_num();
        a[TID] = 0.0; //False sharing
    }//End of parallel sharing
    With each update of "a", the cache line moves to the cache of the thread executing the update!

Homework
    1. Always make a profile before and after (using profiling tool)
    2. Details sometimes make all the difference
    3. In many cases, a performance mystery is explained by NUMA effects, false sharing, or both

How to approach NUMA
    Tuning for NUMA is about keeping threads and their data close
    In OpenMP, a thread may be moved to the data, rather than moving data to threads
    Affinity constructs inOpenMP control where threads run
    This is a powerful OpenMP feature, but it's my responsibility to get right
    So where does data get allocated then?
        Managed by OS. First Touch Placment Policy allocates data page in memory closest to the thread accessing the page for the first time.
        So whoever uses the allocated first owns it.
        Policy is default on Linux and other OSes
    What if single thread initialized most or all data?
        All data ends up in memory of a single node
        Increases access times
    Solution: Parallelize data initiailzation part!
        Example
        #pragma omp parallel for scedule(static)
        for(int i = 0; i<n; i++)
            a[i]=0;
    
    Matrix*Vector test code
    #pragma omp parallel for default(none) shared(m,n,a,b,c) schedule(static)
    for(int i = 0; i < m; i++){
        double sum=0.0;
        for(int j=0; j<n; j++){
            sum += b[i][j]*c[j];
        }
        a[i]=sum
    }
    Anything wrong?
        Runs in parallel
        Data initialization is sequential.
    More NUMA friendly NUMA implementation

    
Question:
How can I dynamically allocate large arrays if I need to be NUMA-aware? Should I use malloc() or would the entire array be placed in one core?
Depends on how to use malloc()

Malloc only requests data. 
Make a large malloc outside parallel region and DON"T touch it
Or initialize in parallel region if possible

Calloc initializes data to a value. It is evil, don't do it unless it's in the 

In C++, how do you use std::vectors in parallel
Probably do it in parallel region if can be used in each core

Other scheduling algorithms
Allocate memory in random threads and hope for the best
*/

conv_ret conv4d_convolve_OpenMP_naive(conv4d_layer layer, featuremap_3d input, featuremap_3d output)
{
    //Benchmarking setup
    conv_ret r;
    time_t start_t, end_t;
    time(&start_t);
    //TODO stub
    //End benchmarking         
    time(&end_t);
    r.time_elapsed = difftime(end_t, start_t);
    return r;
}
conv_ret conv4d_convolve_OpenMP_optimized(conv4d_layer layer, featuremap_3d input, featuremap_3d output, const size_t block_size)
{
    //Benchmarking setup
    conv_ret r;
    time_t start_t, end_t;
    time(&start_t);
    //TODO stub
    //End benchmarking         
    time(&end_t);
    r.time_elapsed = difftime(end_t, start_t);
    return r;
}

#endif