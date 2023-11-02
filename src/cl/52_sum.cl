#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#ifndef __NVCC__ // если это не NVCC компилятор, т.е. если это действительно OpenCL (а не CUDA, т.е. не случай когда нас заинклюдили в .cu файл сразу после инклюда opencl_translator.cu)
    #define STATIC_KEYWORD // говорим что это ключевое слово ничего не должно писать
#endif

#line 6

STATIC_KEYWORD void sum_1(__global const int* a,
                          __global       int* sum,
                                unsigned int  n) {
    const size_t index = get_global_id(0);

    if (index >= n)
        return;

    sum[0] += a[index];
}

STATIC_KEYWORD void sum_2(__global const int* a,
                          __global       int* sum,
                                unsigned int  n) {
    const size_t index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(sum, a[index]);
}

#define VALUES_PER_WORKITEM 64
STATIC_KEYWORD void sum_3(__global const int* a,
                          __global       int* sum,
                                unsigned int  n) {
    const size_t index = get_global_id(0);

    if (index * VALUES_PER_WORKITEM >= n)
        return;

    int mySum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        mySum += a[index * VALUES_PER_WORKITEM + i];
    }

    atomic_add(sum, mySum);
}

STATIC_KEYWORD void sum_3_no_cache_but_coalesced(__global const int *a,
                                                 __global       int *sum,
                                                       unsigned int  n) {
    const size_t index = get_global_id(0);

    const size_t step_size = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM; // это почти что N/64, но с округлением вверх
    if (index >= step_size)
        return;

    int mySum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        mySum += a[index + i * step_size];
    }

    atomic_add(sum, mySum);
}

STATIC_KEYWORD void sum_3_no_cache_no_coalesced(__global const int* a,
                                                __global const int* shuffled_indices,
                                                __global       int* sum,
                                                      unsigned int  n) {
    const size_t index = get_global_id(0);

    const size_t step_size = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM; // это почти что N/64, но с округлением вверх
    if (index >= step_size)
        return;

    size_t random_index = shuffled_indices[index];

    int mySum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        mySum += a[random_index + i * step_size];
    }

    atomic_add(sum, mySum);
}

__attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void sum_gpu(__global const int* a,
                      __global       int* sum,
                            unsigned int  n)
{
    sum_3(a, sum, n);
}

__attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void sum_3_no_cache_no_coalesced_gpu(__global const int* a,
                                              __global const int* shuffled_indices,
                                              __global       int* sum,
                                                    unsigned int  n)
{
    sum_3_no_cache_no_coalesced(a, shuffled_indices, sum, n);
}
