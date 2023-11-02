#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:22
#include "cl/52_sum_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


#ifdef CUDA_SUPPORT
void cuda_sum_gpu(const gpu::WorkSize &workSize,
                  const int* a, int* sum, unsigned int n,
                  cudaStream_t stream);
void cuda_sum_gpu_no_cache_no_coalesced(const gpu::WorkSize &workSize,
                                        const int *a, const int *shuffled_indices, int *sum, unsigned int n,
                                        cudaStream_t stream);
#endif

std::vector<int> generateShuffledIndicies(size_t n)
{
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }
    std::srand(239); // fix seed
    std::random_shuffle(indices.begin(), indices.end());
    return indices;
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    bool is_cuda;
#ifdef CUDA_SUPPORT
    if (device.supports_cuda) {
        context.init(device.device_id_cuda);
        is_cuda = true;
        std::cout << "Using API: CUDA" << std::endl;
    } else
#endif
    {
        context.init(device.device_id_opencl);
        is_cuda = false;
        std::cout << "Using API: OpenCL" << std::endl;
    }
    context.activate();

//    unsigned int n=10*1000*1000;
    unsigned int n=100*1000*1000;
    int max_value = std::numeric_limits<int>::max() / n; // таким образом гарантируем что нет переполнения
    std::vector<int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    #define VALUES_PER_WORKITEM 64
    int step_size = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
    std::vector<int> shuffled_indices = generateShuffledIndicies(step_size);

    gpu::gpu_mem_32i as_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);

    gpu::gpu_mem_32i shuffled_indices_gpu;
    shuffled_indices_gpu.resizeN(shuffled_indices.size());
    shuffled_indices_gpu.writeN(shuffled_indices.data(), shuffled_indices.size());

    gpu::gpu_mem_32i sum_gpu;
    sum_gpu.resizeN(1);
    const int sum_is_zero = 0;

    unsigned int workGroupSize = 128;
    ocl::Kernel opencl_sum_gpu(sum_sources, sum_sources_length, "sum_gpu", "-DWORKGROUP_SIZE=" + to_string(workGroupSize));
    ocl::Kernel opencl_sum_gpu_no_cache_no_coalesced(sum_sources, sum_sources_length, "sum_3_no_cache_no_coalesced_gpu", "-DWORKGROUP_SIZE=" + to_string(workGroupSize));
    if (!is_cuda) {
        opencl_sum_gpu.compile();
        opencl_sum_gpu_no_cache_no_coalesced.compile();
    }

    unsigned int niters = 10;

    timer t;
    for (unsigned int i = 0; i < niters; ++i) {
//        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize; // для sum_1/sum_2
        unsigned int global_work_size = (step_size + workGroupSize - 1) / workGroupSize * workGroupSize; // не забываем исправить размер NDRange для sum_3
        gpu::WorkSize workSize(workGroupSize, global_work_size);
        sum_gpu.writeN(&sum_is_zero, 1);
        if (is_cuda) {
//            cuda_sum_gpu(workSize, as_gpu.cuptr(), sum_gpu.cuptr(), n, context.cudaStream());
            cuda_sum_gpu_no_cache_no_coalesced(workSize, as_gpu.cuptr(), shuffled_indices_gpu.cuptr(), sum_gpu.cuptr(), n, context.cudaStream());
        } else {
//            opencl_sum_gpu.exec(workSize, as_gpu, sum_gpu, n);
            opencl_sum_gpu_no_cache_no_coalesced.exec(workSize, as_gpu, shuffled_indices_gpu, sum_gpu, n);
        }
        t.nextLap();
    }
    std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "Reached memory bandwidth: " << (1.0 * n * sizeof(int) / t.lapAvg() / 1024.0 / 1024.0 / 1024.0) << " GB/s" << std::endl;

    int sum_cpu_results = 0;
    for (int i = 0; i < n; ++i) {
        sum_cpu_results += as[i];
    }

    int sum_gpu_results = -1;
    sum_gpu.readN(&sum_gpu_results, 1);

    EXPECT_THE_SAME(sum_gpu_results, sum_cpu_results, "GPU results should be equal to CPU results!");

    return 0;
}
