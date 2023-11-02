Суммирование элементов массива с локальной редукцией
=========

Давайте посмотрим на кернел суммирования элементов массива - используя локальную редукцию в локальной памяти и барьеры синхронизации:

```c++
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
```

Профилируем
=========

Не забудьте убедиться что вернули объем рабочей нагрузки ```n=100*1000*1000```.

Запустим и удивимся низкой эффективно достигнутиой пропускной способности: **281 GB/s = 28% SOL**.

Запустим под профилировщиком и увидим что вроде бы не должно быть так, вроде бы в первом (единственном долгом) кернеле исполнение достигло **95% ПСП**:

![NVIDIA Nsight Compute summary](/docs/images/nvc_sum_shared_summary.png?raw=true)

А остальные запуски кернела пренебрежимы по своей длительности.

Почему же так? Давайте подробнее взглянем как выглядит общая картина, вдруг мы что-то упускаем.

Визуализация всего Timeline
=========

Для того чтобы построить **Timeline** происходящего в нашем приложении - создаем новый проект, указываем опять исполняемый бинарник и выставляем две галочки на **CUDA** - одну в **Application** вкладке, другую во вкладке **System**:

![NVIDIA Nsight Compute timeline launch](/docs/images/nvc_sum_shared_timeline_launch.png?raw=true)

Теперь мы видим примерно такую картину:

![NVIDIA Nsight Compute timeline overview](/docs/images/nvc_sum_shared_timeline_overview.png?raw=true)

Обратите внимание что большую часть времени тут происходит неинтересное нам. Нас интересует только выделенное красным - только в этом малом окошке выполнялись кернелы.

Давайте выделим это временное окно мышкой, нажмем там правой кнопкой и выберем ```Filter and Zoom in```:

![NVIDIA Nsight Compute after filter and zoom in](/docs/images/nvc_sum_shared_timeline_zoomin_ten_launches.png?raw=true)

Видно 10 очень похожих запусков.

```Упражнение``` Посмотрите в ```.cpp``` коде - что это за десять запусков? 10 редукций?

Теперь давайте выберем любой из запусков и сделаем ```Filter and Zoom in``` в него:

![NVIDIA Nsight Compute memcpy and kernel execution](/docs/images/nvc_sum_shared_timeline_memcpy_and_kernel.png?raw=true)

Заметно что большую часть времени занимало некое копирование памяти, и оно же есть в последующих 9 запусках.

```Упражнение``` Посмотрите в ```.cpp``` коде - какие там есть копирования ```DtoD```? То есть ```Device to Device = VRAM -> VRAM```.

Замечаем что мы ведь должны скопировать буфер с переменными - чтобы редуцирование в этот массив не испортило будущие запуски. Давайте добавим сразу после этого копирования синхронизацию ```CUDA``` чтобы дождаться окончания копирования, а после этого сбросим секундомер:

```c++
    for (unsigned int i = 0; i < niters; ++i) {
        as_gpu.copyToN(bs_gpu, n);
        CUDA_SAFE_CALL(cudaDeviceSynchronize()); // дожидаемся окончания копирования массива
        t.restart(); // сбрасываем секундомер
```

Видим что теперь у нас достигнуто **741 GB/s = 74% SOL**. Посмотрим как теперь выглядит **Timeline**:

![NVIDIA Nsight Compute memcpy with sync and kernel execution](/docs/images/nvc_sum_shared_timeline_memcpy_sync_and_kernel.png?raw=true)

```Упражнение``` Подумайте - что если копировать входные данные из каждый раз? Т.е. в каждой итерации делать ```as_gpu.writeN(as.data(), n);```.

```Упражнение``` Проверьте - сделайте такую копию вместо ```DtoD``` копирования, как изменилась картина Timeline?

Если мы заменим копирование вспомогательного видеобуфера во входной видеобуфер, т.е. ```as_gpu.copyToN(bs_gpu, n);``` на явную прогрузку данных с ЦПУ каждый раз, т.е. напишем ```bs_gpu.writeN(as.data(), n);```.

В таком случае Timeline будет выглядеть следующим образом:

![NVIDIA Nsight Compute pcie bottleneck](/docs/images/nvc_sum_shared_timeline_pcie_and_kernel.png?raw=true)

Явно видно что трансфер данных ```HtoD = Host to Device = RAM -> VRAM``` стал можарировать вообще все, т.к. у PCI-E 3.0 шины показатели например **8 GB/s** (у PCI-E 4.0 в два раза больше) в соответствии с рисунком ниже, а у **VRAM** ПСП может быть **400 GB/s** или например **1008 GB/s**.

![pcie vs vram bandwidth](/docs/images/pcie_vs_vram_bandwidth.png?raw=true)

И это очень важно запомнить - если есть некие данные над которыми нужно выполнить некоторую работу - то работы должно быть **принципиально больше** чем трансфера этих данных на вычислительное устройство! Иначе может быть быстрее обработать на **ЦПУ**.

Любопытный факт - как бы вдруг быстра не стала PCI-E шина, все-равно данные лежат в **RAM**, а значит если вычислительная работа не значительно больше чем объем этих данных - то все еще быстрее обработать на ЦПУ, т.к. ограничивающий фактор в любом случае - это "прочитать данные", а это делается за одинаковое время и в случае ```RAM -> PCI-E -> VRAM -> GPU``` и в случае ```RAM -> CPU```.
