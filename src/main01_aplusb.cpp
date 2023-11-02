#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:22
#include "cl/01_aplusb_cl.h"

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
void cuda_aplusb(const gpu::WorkSize &workSize,
                 const float* a, const float* b, float* c, unsigned int n,
                 cudaStream_t stream);
#else
void cuda_aplusb(const gpu::WorkSize &workSize,
                 const float* a, const float* b, float* c, unsigned int n,
                 cudaStream_t stream)
{
    throw std::runtime_error("CUDA_SUPPORT=OFF!");
}
#endif


int main(int argc, char **argv)
{
    // Это пример использования библиотеки для решения предыдущего задания A+B

    // chooseGPUDevice:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    // Этот контекст после активации будет прозрачно использоваться при всех вызовах в libgpu библиотеке
    // это достигается использованием thread-local переменных, т.е. на самом деле контекст будет активирован для текущего потока исполнения
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

    unsigned int n = 10*1000*1000;
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // Создаем три буфера в видеопамяти
    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    cs_gpu.resizeN(n);

    // Прогружаем данные из векторов as и bs
    // (есть нетипизированный метод write для которого количество измеряется в байтах,
    // и типизированный writeN, для которого количество измеряется в количестве float-элементов, т.к. gpu::gpu_mem_32f - это shared_device_buffer_typed<float>)
    as_gpu.writeN(as.data(), n);
    bs_gpu.writeN(bs.data(), n);

    unsigned int workGroupSize = 128;

    // Исходники кернела написаны в src/cl/aplusb.cl
    // Но благодаря convertIntoHeader(src/cl/aplusb.cl src/cl/aplusb_cl.h aplusb_sources) (см. CMakeLists.txt:18)
    // при компиляции автоматически появится файл src/cl/aplusb_cl.h с массивом aplusb_sources состоящим из байт исходника
    // т.о. программе не будет нужно в runtime читать файл с диска, т.к. исходник кернелов теперь хранится в массиве данных основной программы
    ocl::Kernel opencl_aplusb(aplusb_sources, aplusb_sources_length, "aplusb", "-DWORKGROUP_SIZE=" + to_string(workGroupSize));
    if (!is_cuda) {
        opencl_aplusb.compile();
    }

    unsigned int niters = 10;

    timer t;
    for (unsigned int i = 0; i < niters; ++i) {
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu::WorkSize workSize(workGroupSize, global_work_size);
        if (is_cuda) {
            cuda_aplusb(workSize, as_gpu.cuptr(), bs_gpu.cuptr(), cs_gpu.cuptr(), n, context.cudaStream());
        } else {
            opencl_aplusb.exec(workSize, as_gpu, bs_gpu, cs_gpu, n);
        }
        t.nextLap();
    }
    std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "Reached memory bandwidth: " << (3.0 * n * sizeof(float) / t.lapAvg() / 1024.0 / 1024.0 / 1024.0) << " GB/s" << std::endl;

    cs_gpu.readN(cs.data(), n);

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(cs[i], as[i] + bs[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
