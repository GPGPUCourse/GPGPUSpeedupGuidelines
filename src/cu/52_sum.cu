#include <libgpu/cuda/cu/opencl_translator.cu>

#define WORKGROUP_SIZE 128

#include "../cl/52_sum.cl"

void cuda_sum_gpu(const gpu::WorkSize &workSize,
                  const int* a, int* sum, unsigned int n,
                  cudaStream_t stream)
{
    sum_gpu<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a, sum, n);
    CUDA_CHECK_KERNEL(stream);
}

void cuda_sum_gpu_no_cache_no_coalesced(const gpu::WorkSize &workSize,
                                        const int *a, const int *shuffled_indices, int *sum, unsigned int n,
                                        cudaStream_t stream)
{
    sum_3_no_cache_no_coalesced_gpu<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a, shuffled_indices, sum, n);
    CUDA_CHECK_KERNEL(stream);
}
