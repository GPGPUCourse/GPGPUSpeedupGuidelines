#include <libgpu/cuda/cu/opencl_translator.cu>

#define WORKGROUP_SIZE 128

#include "../cl/01_aplusb.cl"

void cuda_aplusb(const gpu::WorkSize &workSize,
                 const float* a, const float* b, float* c, unsigned int n,
                 cudaStream_t stream)
{
    aplusb<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a, b, c, n);
    CUDA_CHECK_KERNEL(stream);
}
