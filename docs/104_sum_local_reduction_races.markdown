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

Проверим что мы можем легко ловить гонки
=========

Запустим под ```compute-sanitizer --tool racecheck``` (бывший ```cuda-memcheck```):

```
sudo /usr/local/cuda/bin/compute-sanitizer --tool racecheck .../cmake-build-relwithdebinfo/main53_sum_local_reduction
```

И... оно работает очень долго. Давайте уменьшим объем нагрузки ```n=100*1000*1000``` в сто раз (не забудьте потом вернуть обратно) и перезапустим:

```
========= COMPUTE-SANITIZER
OpenCL devices:
  Device #0: GPU. NVIDIA GeForce RTX 4090 (CUDA 12010). Free memory: 22753/24214 Mb
Using device #0: GPU. NVIDIA GeForce RTX 4090 (CUDA 12010). Free memory: 22783/24214 Mb
Using API: CUDA
Data generated for n=1000000!
GPU: 0.205047+-0.000380515 s
Reached memory bandwidth: 0.018168 GB/s
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

Хорошо, давайте что-нибудь сломаем - закомментируем барьер в кернеле:

```c++
int workgroup_sum = 0;
//barrier(CLK_LOCAL_MEM_FENCE); // БАРЬЕР КОТОРЫЙ МЫ ЗАКОММЕНТИРОВАЛИ
if (lid == 0) {
```

Затем перекомпилируем и запустим вновь ```compute-sanitizer```:

![NVIDIA Nsight Compute example of race condition](/docs/images/nvc_sum_shared_example_of_race_condition.png?raw=true)

Номера строк в которых произошло обращение возможно было восстановить благодаря флагу ```-lineinfo``` переданному компилятору ```NVCC``` в ```CMakeLists.txt```.

Итого видно что случилась гонка между **записью в 22 строке** и **чтением в 35 строке** (иногда такую гонку называют **RAW = Read After Write**):

```c++
    buf[lid] = 0;
    if (gid < step_size) {
        for (int i = 0; i < WORKITEM_K; ++i) {
            size_t index = gid + step_size * i;
            buf[lid] += index < n ? a[index] : 0; // здесь запись (в buf[lid])
        }
    }

    int workgroup_sum = 0;
    //barrier(CLK_LOCAL_MEM_FENCE); // барьер который мы закомментировали
    if (lid == 0) {
        for (int i = 0; i < WORKGROUP_SIZE; ++i) {
            workgroup_sum += buf[i]; // здесь чтение (из buf[i])
        }
```

Видим что действительно - т.к. чтение иногда выполняется по адресам по которым писали другие потоки рабочей группы - то необходимо провести синхронизацию, чтобы достоверно дождаться что остальные потоки закончили свои записи, и мы можем спокойно увидеть результаты их работы. Для этого здесь и был написан изначально барьер.

```Упражнение``` А как надо изменить этот кернел (возможно его другую версию) чтобы спровоцировать **WAR = Write After Read** тип гонки?

Попробуем спровоцировать **WAR** тип гонки:

```c++
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nvalues = WORKGROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * lid < nvalues) {
            int v1 = buf[lid];
            int v2 = buf[lid + nvalues / 2]; // 43 строка (чтение)
            buf[lid] = v1 + v2;              // 44 строка (запись)
        }
        //barrier(CLK_LOCAL_MEM_FENCE); // барьер который мы закомментировали
    }
```

И действительно одна из обнаруженных ошибок будет **WAR = Write After Read** в 43 и 44 строке:

```
========= Warning: Race reported between Read access at 0x300 in //cl/53_sum_local_reduction.cl:43:sum_local_reduction(const int *, int *, unsigned int, unsigned int)
=========     and Write access at 0x320 in //cl/53_sum_local_reduction.cl:44:sum_local_reduction(const int *, int *, unsigned int, unsigned int) [8 hazards]
```

```Упражнение``` Но как такое возможно? Ведь казалось бы, если поток в 43 строке прочитал по одному адресу, то в 44 строке он спокойно запишет в ячейку по другому адресу?

```Упражнение``` Мы встретили **RAW** и **WAR** типы гонок. Какие еще могут быть в теории? Можно ли придумать случай когда можно случайно сделать такую багу?
