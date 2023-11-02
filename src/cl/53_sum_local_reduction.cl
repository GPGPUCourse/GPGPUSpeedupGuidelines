#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__attribute__((reqd_work_group_size(WORKGROUP_SIZE, 1, 1)))
__kernel void sum_local_reduction(__global const int* a,
                                  __global       int* sum,
                                        unsigned int  step_size,
                                        unsigned int  n) {
    const size_t gid = get_global_id(0);
    const size_t lid = get_local_id(0);
    const size_t wid = get_group_id(0);

    __local int buf[WORKGROUP_SIZE];

    buf[lid] = 0;
    if (gid < step_size) {
        for (int i = 0; i < WORKITEM_K; ++i) {
            size_t index = gid + step_size * i;
            buf[lid] += index < n ? a[index] : 0;
        }
    }

#define LOCAL_REDUCTION_MASTER 1
#define LOCAL_REDUCTION_TREE   2
#define LOCAL_REDUCTION_MODE   LOCAL_REDUCTION_MASTER

    int workgroup_sum = 0;
#if LOCAL_REDUCTION_MODE == LOCAL_REDUCTION_MASTER
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        for (int i = 0; i < WORKGROUP_SIZE; ++i) {
            workgroup_sum += buf[i];
        }
    }
#elif LOCAL_REDUCTION_MODE == LOCAL_REDUCTION_TREE
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = WORKGROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * lid < nvalues) {
            int v1 = buf[lid];
            int v2 = buf[lid + nvalues / 2];
            buf[lid] = v1 + v2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    workgroup_sum = buf[0];
#endif

    if (lid == 0) { // если мы мастер-поток в своей рабочей группе
        sum[wid] = workgroup_sum;
    }
}