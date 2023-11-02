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

Проверим что мы можем проверять на Out-of-Bounds
=========

Иногда можно случайно обратиться по неправильному адресу памяти и получить неверный результат или падение кернела с ```an illegal memory access was encountered```.

Давайте симулируем такую ситуацию и проверим что ```compute-sanitizer``` с этим успешно справляется:

Заменим эту строку

```c++
buf[lid] += index < n ? a[index] : 0;
```

на вот такую:

```c++
buf[lid] += index < n ? a[index - 1] : 0;
```

И давайте уменьшим объем нагрузки ```n=100*1000*1000``` в сто раз (не забудьте потом вернуть обратно) чтобы ускорить проверку. После изменения не забудьте перекомпилировать.

Запускаем ```compute-sanitizer --tool memcheck```:

```sudo /usr/local/cuda/bin/compute-sanitizer --tool memcheck /.../main53_sum_local_reduction```

Получаем:

![NVIDIA Nsight Compute example of index out of bounds](/docs/images/nvc_sum_shared_example_of_index_out_of_bounds.png?raw=true)

Здесь несколько интересных деталей:

- Указан размер считанного значения (4 байта)
- Указана строка в кернеле которая произвела некорректное обращение к памяти (22 строка)
- Указано какой поток (local id) и какой рабочей группы (wokrgroup id) это сделал
- Указано даже что ближайшая аллокация в памяти по такому-то адресу и такого-то размера (4 мегабайта)
