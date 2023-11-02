Пререквизиты
=========

Установлена **CUDA** (при написании этих заметок использовалась **CUDA 12.1** на **Ubuntu 22.04** с NVIDIA драйвером **530.30.02**).

Как установить **NVIDIA драйвер**
=========

```WARNING``` Все что связано с установкой/удалением видеодрайверов нужно делать с опаской - все что угодно может пойти не так и у вас может не загрузиться GUI операционной системы. Поэтому не делайте этого если у вас нет в запасе несколько часов/дней/лет для того чтобы если что-то пойдет не так - провести воскрешение видеодрайвера и GUI.

Предполагается что у вас чистый дистрибутив без видеодрайвера. Если вы уже устанавливали видеодрайвер через ```sudo apt install nvidia-NNN``` - сначала его надо удалить через ```sudo apt purge nvidia-*```.

Скачиваем и устанавливаем драйвер:

```
NVIDIA_DRIVER=530.30.02
NVIDIA_DRIVER_URL=https://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER}.run
wget ${NVIDIA_DRIVER_URL}
chmod +x NVIDIA-Linux-x86_64-${NVIDIA_DRIVER}.run
sudo ./NVIDIA-Linux-x86_64-${NVIDIA_DRIVER}.run --no-questions --accept-license --no-precompiled-interface --ui=none
```

Как установить **CUDA 12.1**
=========

```WARNING``` Все что связано с установкой/удалением видеодрайверов нужно делать с опаской - все что угодно может пойти не так и у вас может не загрузиться GUI операционной системы. Поэтому не делайте этого если у вас нет в запасе несколько часов для того чтобы если что-то пойдет не так - провести воскрешение видеодрайвера и GUI.

Такая установка CUDA **не будет** устанавливать видеодрайвер который обычно идет с ней в комплекте. Но это может ограничить работоспособность профилировщика если видеодрайвер более свежий чем CUDA SDK. Поэтому в данной инструкции выше предлагается установить видеодрайвер совпадающий с видеодрайвером поставляемым с CUDA SDK (в целом его можно и в рамках установки CUDA SDK установить).

Скачиваем и устанавливаем CUDA SDK:

```
CUDA_RUNFILE=cuda_12.1.1_530.30.02_linux.run
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/${CUDA_RUNFILE}
sudo apt-get install linux-headers-$(uname -r)
chmod +x ${CUDA_RUNFILE}
sudo ./${CUDA_RUNFILE} --silent --toolkit
# permamently add NVCC to PATH:
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
echo "export PATH=\$PATH:\$CUDA_HOME/bin" >> ~/.bashrc
```

Запуск NVIDIA Nsight Compute
=========

Запускаем ```/usr/local/cuda/nsight-compute-2023.1.1/ncu-ui``` и создаем новый проект (нужно указать бинарный файл нашего приложения и файл куда будет сохранен результат профилироования):

![NVIDIA Nsight Compute new project](/docs/images/nvc_new_project.png?raw=true)

Если вы видите ошибку как на картинке ниже, то открываем [ссылку](https://developer.nvidia.com/ERR_NVGPUCTRPERM) как сказано и выполняем все по инструкции, но самый простой способ это просто перезапустить через ```sudo /usr/local/cuda/nsight-compute-2023.1.1/ncu-ui```.

![NVIDIA Nsight Compute ERR_NVGPUCTRPERM The user does not have permission to access NVIDIA GPU Performance Counters](/docs/images/nvc_no_permission_error.png?raw=true)

После перезапуска из-под sudo и создания нового проекта (указав бинарный файл нашего приложения и файл куда сохранять результаты профилирования) мы получаем **Summary**:

![NVIDIA Nsight Compute summary example](/docs/images/nvc_aplusb_summary.png?raw=true)

Из любопытного - можно заметить что **Memory Throughput** почти полностью использует возможности видеокарты, т.е. мы упираемся в пропускную способность (**ПСП**) видеопамяти.

Давайте теперь дважды кликнем по цифре **2** в колонке **Issues Detected** и увидим что в целом у нас нет каких-то очевидных проблем (что ожидаемо ведь кернел довольно примитивный):

![NVIDIA Nsight Compute details failed to find section memory workload analysis](/docs/images/nvc_aplusb_details_failed_to_find_section_memory_workload_analysis.png?raw=true)

И чтобы взглянуть на то как выглядит **Memory Workload Analysis** - оказывается надо было включить сбор соответствующих метрик, поэтому заново создаем чистый проект для запуска и включаем все метрики:

![NVIDIA Nsight Compute enable all profiling metrics](/docs/images/nvc_aplusb_enabling_all_metrics.png?raw=true)

На этот раз мы видим аж **8 Issues Detected** и если сделать по ним двойной клик то в **Details** из любопытного мы увидим например в **Memory Workload Analysis**:

![NVIDIA Nsight Compute Memory Workload Analysis](/docs/images/nvc_aplusb_memory_rw_size.png?raw=true)

Видно что всего было считано 80 мегабайт, а записано 40 мегабайт. Это сходится с ожиданием, ведь у нас 10^7 float элементов в двух суммируемых массивах (т.е. каждый как раз 40 мегабайт, итого считывается 80 мегабайт). И результат суммирования пишется в третий массив который как раз 40 мегабайт.

И так же любопытно посмотреть на работу с видеопамятью с точки зрения пропускной способности памяти (**ПСП**), она оказывается насыщена больше чем на 95%:

![NVIDIA Nsight Compute Memory Workload Analysis Throughput](/docs/images/nvc_aplusb_memory_throughput.png?raw=true)
