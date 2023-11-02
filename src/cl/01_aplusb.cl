#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void aplusb(__global const float* a,
                     __global const float* b,
                     __global       float* c,
                     unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    c[index] = a[index] + b[index];
}
