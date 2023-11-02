#include <libgpu/cuda/cu/opencl_translator.cu>

#define WORKGROUP_SIZE 128
#define WORKITEM_K     1

#include "../cl/53_sum_local_reduction.cl"

void cuda_sum_local_reduction_gpu(const gpu::WorkSize &workSize,
                                  const int* a, int* sum, unsigned int step_size, unsigned int n,
                                  cudaStream_t stream)
{
    sum_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a, sum, step_size, n);
    CUDA_CHECK_KERNEL(stream);
}
