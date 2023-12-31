[Остальные задания](https://github.com/GPGPUCourse/).

[![Build Status](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/actions/workflows/cmake.yml/badge.svg?branch=main&event=push)](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/actions/workflows/cmake.yml)

# Видеозапись лекция (две части)

<a href="https://www.youtube.com/watch?v=hqPvJ0sa_gQ"><img src="https://raw.githubusercontent.com/GPGPUCourse/GPGPUSpeedupGuidelines/master/.github/video_preview1.png" alt="Presentation on ICCV 2021" width="32%"/></a> <a href="https://www.youtube.com/watch?v=qnbxOv4xZcU"><img src="https://raw.githubusercontent.com/GPGPUCourse/GPGPUSpeedupGuidelines/master/.github/video_preview2.png" alt="Presentation on ICCV 2021" width="32%"/></a>

# Используем профилировщик/санитайзер

Первая лекция:
1) [Статья 01:](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/101_how_to_profile.markdown) как установить NVIDIA драйвер и CUDA, пример запуска профилировщика **NVIDIA Nsight Compute** для задачи суммирования двух векторов
2) [Статья 02:](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/102_how_to_sanitize.markdown) как запустить санитайзер **compute-sanitizer --tool initcheck** для проверки что вся видеопамять была инициализирована (не считываем случайный мусор)

Вторая лекция:

3) [Статья 03:](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/103_sum.markdown) на примере задачи суммирования элементов массива исследуем и профилируем:
- 3.1) Как отладить ошибку через ```printf``` со стороны кернела на видеокарте
- 3.2) Иногда кэш спасает скорость работы кернела вопреки **non-coalesced** паттерну доступа
- 3.3) Как помешать кэшу спасать нас
- 3.4) Как форсировать чтения памяти быть **non-coalesced**
4) [Статья 04:](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/104_sum_local_reduction_races.markdown) на примере задачи суммирования элементов массива через редукцию в локальной памяти:
- 4.1) Как проверить нет ли в кернеле гонок в обращениях к локальной памяти (**compute-sanitizer --tool racecheck**)
- 4.2) Какие есть виды гонок (на примере **RAW**, **WAR** гонок)
5) [Статья 05:](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/105_sum_local_reduction_out_of_bounds.markdown) на примере задачи суммирования элементов массива через редукцию проверяем что обращение за пределами массива легко поймать (**compute-sanitizer --tool memcheck**)
6) [Статья 06:](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/106_sum_local_reduction_profiling.markdown) на примере задачи суммирования элементов массива через редукцию анализируем поведение программы через **Timeline** визуализацию в  **NVIDIA Nsight Compute**

# Примеры профилирования/ускорения

11) **[Пример](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/111_vulkan_slow_vram_to_ram.markdown) профилирования и ускорения:** случай когда тормозит трансфер данных на Vulkan (```VRAM -> CPU```) - помогает увидеть timeline + видеть в логе строку вида ```processing done in 1234 s = 20% IO + 10% CPU + 5% upload to VRAM + 30% GPU + 35% read from VRAM``` + взвешивать килограммами

12) **[Пример](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/112_two_threads.markdown) профилирования и ускорения:** в случае когда CPU-часть занимает существенную долю времени (например 40%) - какой потенциальный прирост от обработки в два потока? А если ситуация другая - 80% времени CPU и 20% GPU?  Что с этим делать?

13) **[Пример](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/113_camera_to_world_to_camera.markdown) профилирования и ускорения:** если есть переход из локальной системы координат **камеры A** в **мир**, и затем из **мира** в локальную систему координат второй **камеры B**

14) **[Пример](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/docs/114_transform_approximation.markdown) профилирования и ускорения:** если какой-то кусок-патч картинки нужно целиком подвергнуть сложной трансформации - вместо проецирования каждого пикселя этого патча - проецируем центр и производные рассчитываем, получили чуть загрубленую трансформацию

# Примеры общего подхода

21) **Пример подхода:** пусть обработка выглядит как IO -> CPU -> GPU, как решить сколько потоков нужно на IO? Сколько на CPU?

22) **Пример подхода:** если есть рандомность в алгоритме - как реализовать ее в модели массового параллелизма? А если хочется детерминизма результата?

23) **Пример подхода:** как в растеризационном пайплайне рассчитать площадь треугольников измеряемых в количестве их пикселей-фрагментов (которые выжили после depth-test в фрагментном шейдере)?

24) **Пример подхода:** как распределить по узлам кластера задачу "для каждого треугольника перечислить перечен камер которые его видят". А что если треугольники не влезают в память? Как сделать это с out-of-core свойством?

25) **Пример подхода:** если надо писать что-то многопоточно в пиксели картинки, как это делать без точки синхронизации, т.е. без одного общего лока? (случай CPU, случай GPU, случай 64-битных данных, случай произвольных по размеру)

26) **Пример подхода:** если задача на некоторой 2D решетке (например картинка) и не требует абсолютной точности - то можно часто прорядить и работать в каждом втором ряду-столбике. А потом несколько итераций уточнить ответ уже для каждого пикселя.

27) **Пример подхода:** если задача на некоторой 2D решетке (например картинка) и не требует абсолютной точности - то можно сначала работать на х32 раза меньшей детальности, затем на х16, ..., наконец, на оригинальном разрешении. Очень много общего с предыдущей идеей про прореживание.

28) **Пример подхода:** как ускорить растеризацию треугольников?

# Список тем: примеры алгоритмов

31) **Пример алгоритма:** пусть на видеокарте есть code divergence в coarse-to-fine схеме в задаче подобной приложению чартов в атлас. Как оставить схему но получить ускорение разобравшись с **code divergence**?

32) **Пример алгоритма:** пусть на CPU есть алгоритм **СНМ** (система непересекающихся множеств) с тяжелыми inplace вычислениями "кто из соседних пикселей действительно сосед - т.е. дорогая проверка стоит ли провести с ним union". Как ускорить за счет делегирования на видеокарту проверку "соседства" оставив СНМ-операции на процессоре?

33) **Пример алгоритма:** пусть на CPU есть алгоритм **СНМ** (система непересекающихся множеств), как ускорить за счет многопоточности?

34) **Пример алгоритма:** если есть алгоритм в котором трассировка лучей, как его ускорить на видеокарте (возможно с потерей точности)? Как переделать алгоритм в котором есть не разбиваемая задача "трассируй луч перешагивая по ячейкам пространства" в парадигму "есть N задач каждая O(1)".

# Почему в этом репозитории и OpenCL, и CUDA?

Этот проект иллюстрирует как написать код для видеокарты посредством OpenCL и затем скомпилировать его в т.ч. для исполнения через CUDA. Это дает замечательную возможность использовать инструментарий CUDA:

 - профилировщик - **NVIDIA Nsight Compute**: позволяет посмотреть timeline выполнения кернелов, насколько какой кернел насытил пропускную способность видеопамяти/локальной памяти или ALU, число используемых регистров и локальной памяти (и соответственно насколько высока occupancy), а так же он часто явно подскажет в чем может быть основная потеря скорости работы кернела
 - санитайзер - **compute-sanitizer --tool memcheck** (бывший **cuda-memcheck**): позволяет проверить что нет out-of-bounds обращений к памяти (если есть - укажет проблемную строку в кернеле)
 - санитайзер - **compute-sanitizer --tool racecheck** (бывший **cuda-memcheck**):  позволяет проверить что нет гонок между потоками рабочей группы при обращении к локальной памяти (т.е. что нигде не забыты барьеры). Если гонка есть - укажет на ее характер (**RAW/WAR/WAW**) и на строки в кернеле (обеих операций участвующих в гонке)

# Ориентиры

Здесь предложена трансляция в CUDA на примере задачи C=A+B, трансляция не исчерпывающая, но во многих простых случаях должна работать.
Она реализована благодаря файлу с макросами которые транслируют OpenCL вызовы в CUDA вызовы - [libs/gpu/libgpu/cuda/cu/opencl_translator.cu](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/libs/gpu/libgpu/cuda/cu/opencl_translator.cu).

Дополнительные ориентиры:

 - [CMakeLists.txt](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/CMakeLists.txt#L29-L34): Поиск CUDA-компилятора, добавление для NVCC компилятора флажка 'сохранять номера строк' (нужно чтобы cuda-memcheck мог указывать номера строк с ошибками), компиляция через ```cuda_add_executable```.
 - [aplusb.cu](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/src/cu/01_aplusb.cu): CUDA-кернел транслируется из OpenCL-кернела посредством [макросов](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/libs/gpu/libgpu/cuda/cu/opencl_translator.cu), вызов кернела через функцию ```cuda_aplusb```
 - **main01_aplusb.cpp**: [декларация](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/src/main01_aplusb.cpp#L28-L30) функции ```cuda_aplusb```, [инициализация](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/src/main01_aplusb.cpp#L59) CUDA-контекста, [вызов](https://github.com/GPGPUCourse/GPGPUSpeedupGuidelines/blob/main/src/main01_aplusb.cpp#L112) функции вызывающий кернел
 - [Запись лекции](https://youtu.be/REKRZavy0_s?t=3176) с одного из прошлых прочтений - есть пояснения про этот транслятор и про отличия CUDA
