Чтобы просто и надежно проверять проблемы вроде неинициализированной памяти или гонки при работе с памятью, удобно использовать автоматизированный проверяльщик, раньше в **CUDA** он назывался **cuda-memcheck**, но начиная с **CUDA 12** он был переименован и теперь он называется **Compute Sanitizer**.

Запускаем Compute Sanitizer
=========

Давайте сначала запустим его и взглянем на аргументы:

```/usr/local/cuda/bin/compute-sanitizer --help```

```
NVIDIA (R) Compute Sanitizer
Copyright (c) 2020-2023 NVIDIA Corporation
Version 2023.1.1
Usage: compute-sanitizer [options] [your-program] [your-program-options]

General options:
  -h [ --help ]                         Produce this help message.
  -v [ --version ]                      Print the version number.
  ...                                   ...
  --tool arg (=memcheck)                Set the tool to use.
                                        memcheck  : Memory access checking
                                        racecheck : Shared memory hazard checking
                                        synccheck : Synchronization checking
                                        initcheck : Global memory initialization checking
  ...                                   ...

Memcheck-specific options:
  ...                                   ...

Racecheck-specific options:
  ...                                   ...

Initcheck-specific options:
  ...                                   ...

Synccheck-specific options:
  ...                                   ...

Please see the compute-sanitizer manual for more information.
```

Самый важный параметр - это ```--tool``` т.к. он определяет какого рода проблемы отслеживает инструмент.

compute-sanitizer --tool initcheck
=========

Давайте проверим как хорошо он обнаружает неинициализированную память - закомментируем инициализацию аккумулятора суммы:

```c++
gpu::gpu_mem_32i sum_gpu;
int sum_is_zero = 0;
sum_gpu.resizeN(1);
// sum_gpu.writeN(&sum_is_zero, 1);
```

```
sudo /usr/local/cuda/bin/compute-sanitizer --tool initcheck .../cmake-build-relwithdebinfo/main52_sum
```

Мы видим много подобных ошибок:

```
========= Uninitialized __global__ memory read of size 4 bytes
=========     at 0x100 in /home/polarnick/coding/forks/GPGPUTasks2023/src/cl/52_sum.cl:30:sum_gpu_2(const int *, int *, unsigned int)
=========     by thread (98,0,0) in block (194,0,0)
=========     Address 0x7fbbca800000
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x32e990]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========     Host Frame:__cudart800 [0x2854b]
=========                in /home/polarnick/coding/forks/GPGPUTasks2023/cmake-build-relwithdebinfo/main52_sum
=========     Host Frame:cudaLaunchKernel [0x846eb]
=========                in /home/polarnick/coding/forks/GPGPUTasks2023/cmake-build-relwithdebinfo/main52_sum
=========     Host Frame:/tmp/tmpxft_0000ba61_00000000-6_52_sum.cudafe1.stub.c:1:__device_stub__Z9sum_gpu_2PKiPij(int const*, int*, unsigned int) [0x21a84]
=========                in /home/polarnick/coding/forks/GPGPUTasks2023/cmake-build-relwithdebinfo/main52_sum
=========     Host Frame:/home/polarnick/coding/forks/GPGPUTasks2023/src/cu/52_sum.cu:19:cuda_sum_gpu_2(gpu::WorkSize const&, int const*, int*, unsigned int, CUstream_st*) [0x221ae]
=========                in /home/polarnick/coding/forks/GPGPUTasks2023/cmake-build-relwithdebinfo/main52_sum
=========     Host Frame:/home/polarnick/coding/forks/GPGPUTasks2023/src/main52_sum.cpp:92:main [0x2044d]
=========                in /home/polarnick/coding/forks/GPGPUTasks2023/cmake-build-relwithdebinfo/main52_sum
=========     Host Frame:../sysdeps/nptl/libc_start_call_main.h:58:__libc_start_call_main [0x29d90]
=========                in /lib/x86_64-linux-gnu/libc.so.6
=========     Host Frame:../csu/libc-start.c:379:__libc_start_main [0x29e40]
=========                in /lib/x86_64-linux-gnu/libc.so.6
=========     Host Frame:_start [0x21565]
=========                in /home/polarnick/coding/forks/GPGPUTasks2023/cmake-build-relwithdebinfo/main52_sum
```

Обратите внимание что указывается в т.ч. строка где произошла проблема. Это возможно благодаря явному указанию ```-lineinfo``` в ```CMakeLists.txt```:

```
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo)
```

Благодаря ему после компиляции в ассемблере остаются указания к какой строке исходника относится каждая ассемблерная инструкция, что позволяет обнаружив ошибку ассемблерной инструкции - "вспомнить" из какой строки исходника эта ассемблерная инструкция была сгенерирована.


