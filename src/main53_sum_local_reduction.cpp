#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:22
#include "cl/53_sum_local_reduction_cl.h"

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
#include <libgpu/cuda/utils.h>
void cuda_sum_local_reduction_gpu(const gpu::WorkSize &workSize,
                                  const int* a, int* sum, unsigned int step_size, unsigned int n,
                                  cudaStream_t stream);
#endif

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

    unsigned int n=1*1000*1000;
//    unsigned int n=100*1000*1000;
    int max_value = std::numeric_limits<int>::max() / n; // таким образом гарантируем что нет переполнения
    std::vector<int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    gpu::gpu_mem_32i as_gpu;
    as_gpu.resizeN(n);
    as_gpu.writeN(as.data(), n);

    gpu::gpu_mem_32i bs_gpu;
    bs_gpu.resizeN(n);

    const unsigned int workGroupSize = 128;
    const unsigned int workItemK = 1; // значений на один work-item
    const unsigned int workGroupK = workGroupSize * workItemK;

    gpu::gpu_mem_32i cs_gpu;
    cs_gpu.resizeN((n + workGroupK - 1) / workGroupK);

    ocl::Kernel opencl_sum_local_reduction_gpu(sum_local_reduction_sources, sum_local_reduction_sources_length, "sum_local_reduction",
                                               "-DWORKGROUP_SIZE=" + to_string(workGroupSize) + " -DWORKITEM_K=" + to_string(workItemK));
    if (!is_cuda) {
        opencl_sum_local_reduction_gpu.compile();
    }

    unsigned int niters = 10;

    int sum_cpu_results = 0;
    for (int i = 0; i < n; ++i) {
        sum_cpu_results += as[i];
    }

    timer t;
    for (unsigned int i = 0; i < niters; ++i) {
        as_gpu.copyToN(bs_gpu, n);
//        CUDA_SAFE_CALL(cudaDeviceSynchronize()); // дожидаемся окончания копирования массива
//        t.restart(); // сбрасываем секундомер

        gpu::gpu_mem_32i input_buffer = bs_gpu;
        gpu::gpu_mem_32i output_buffer = cs_gpu;
        unsigned int input_n = n;
        while (input_n > 1) {
            unsigned int output_n = (input_n + workGroupK - 1) / workGroupK;
            gpu::WorkSize workSize(workGroupSize, output_n * workGroupSize);
            unsigned int step_size = (input_n + workItemK - 1) / workItemK;
            if (is_cuda) {
                cuda_sum_local_reduction_gpu(workSize, input_buffer.cuptr(), output_buffer.cuptr(), step_size, input_n, context.cudaStream());
            } else {
                opencl_sum_local_reduction_gpu.exec(workSize, input_buffer, output_buffer, step_size, input_n);
            }
            std::swap(input_buffer, output_buffer);
            input_n = output_n;
        }
        int sum_gpu_results = -1;
        input_buffer.readN(&sum_gpu_results, 1);
        EXPECT_THE_SAME(sum_gpu_results, sum_cpu_results, "GPU results should be equal to CPU results!");

        t.nextLap();
    }
    std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "Reached memory bandwidth: " << (1.0 * n * sizeof(int) / t.lapAvg() / 1024.0 / 1024.0 / 1024.0) << " GB/s" << std::endl;

    return 0;
}
