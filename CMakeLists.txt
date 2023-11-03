cmake_minimum_required(VERSION 3.1)

option(GPU_CUDA_SUPPORT "CUDA support." ON)

add_subdirectory(libs)

project(GPGPUSpeedupGuidelines)

set(CMAKE_CXX_STANDARD 11)

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found!")
endif()

# convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71
# Она считывает все байты из файла src/cl/aplusb.cl (т.е. весь исходный код кернела) и преобразует их в массив байтов в файле src/cl/aplusb_cl.h aplusb_kernel
# Обратите внимание что это происходит на этапе компиляции, кроме того необходимо чтобы файл src/cl/aplusb_cl.h был перечислен среди исходников для компиляции при вызове add_executable
convertIntoHeader(src/cl/01_aplusb.cl src/cl/01_aplusb_cl.h aplusb_sources)
convertIntoHeader(src/cl/52_sum.cl src/cl/52_sum_cl.h sum_sources)
convertIntoHeader(src/cl/53_sum_local_reduction.cl src/cl/53_sum_local_reduction_cl.h sum_local_reduction_sources)

if (GPU_CUDA_SUPPORT)
    find_package(CUDA REQUIRED)
    add_definitions(-DCUDA_SUPPORT)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo)
    cuda_add_executable(main01_aplusb src/main01_aplusb.cpp src/cl/01_aplusb_cl.h src/cu/01_aplusb.cu)
    cuda_add_executable(main52_sum src/main52_sum.cpp src/cl/52_sum_cl.h src/cu/52_sum.cu)
    cuda_add_executable(main53_sum_local_reduction src/main53_sum_local_reduction.cpp src/cl/53_sum_local_reduction_cl.h src/cu/53_sum_local_reduction.cu)
else()
    message(WARNING "CUDA not found!")
    add_executable(main01_aplusb src/main01_aplusb.cpp src/cl/01_aplusb_cl.h)
    add_executable(main52_sum src/main52_sum.cpp src/cl/52_sum_cl.h)
    add_executable(main53_sum_local_reduction src/main53_sum_local_reduction.cpp src/cl/53_sum_local_reduction_cl.h)
endif()

target_link_libraries(main01_aplusb libclew libgpu libutils)
target_link_libraries(main52_sum libclew libgpu libutils)
target_link_libraries(main53_sum_local_reduction libclew libgpu libutils)
