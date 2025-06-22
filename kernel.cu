
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cmath>
#include <fstream>
#include <string>


__global__ void akf_kernel(int* dev_max, size_t* dev_bestSignals, size_t n,size_t N)
{

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int* akf = new int[n];
   size_t loc_bestSignal=0;
   int pr_dev_max = 100000000;
   size_t mxidx = 1;
   mxidx <<= n;
   int max1_val;
   int max2_val;
   size_t k;
   size_t ind;
   size_t i;
   size_t j;
   for (k = idx, ind=0; ind < mxidx/N;k+=N,ind++)
   {
       for (i = 0; i < n; i++) {
           akf[i] = 0;
           for (j = 0; j < n; j++) {
               if (i + j < n) {
                   akf[i] += ((k>> (i + j) & 1) ? 1 : -1) * ((k >> (j) & 1) ? 1 : -1);
               }
           }
       }
       max1_val = -10000000;
       max2_val = -10000000;

       for (i = 0; i < n; ++i) {
           if (abs(akf[i]) > max1_val) {
               max2_val = max1_val;
               max1_val = abs(akf[i]);
           }
           else if (abs(akf[i]) > max2_val && abs(akf[i]) != max1_val) {
               max2_val = abs(akf[i]);
           }
       }
       if (max2_val < pr_dev_max)
       {
           pr_dev_max = max2_val;
           loc_bestSignal = k;
       }
      
   }
   dev_bestSignals[idx] = loc_bestSignal;
   dev_max[idx] = pr_dev_max;
   delete[] akf;
}

__global__ void max_akf_kernel(int* dev_max, size_t n, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    if (idx >= n){
        return;
    }

    int local_result = 0;

        for (size_t j = 0; j <= n; j++) {
            if ((tid + j < n) && (threadIdx.x!=0)) {
                local_result += (((blockIdx.x*N) >> (tid + j) & 1) ? 1 : -1) *
                    (((blockIdx.x*N) >> (j) & 1) ? 1 : -1);
            }
        }



    atomicMax(&dev_max[blockIdx.x], abs(local_result));
}

__global__ void find_minimum_element(int* dev_max, size_t* result_signal, size_t array_length)
{
    extern __shared__ int temp[];  // Разделяемая память для временного хранения
    size_t tid = threadIdx.x;      // Индекс нити в блоке
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  // Глобальный индекс нити

    // Шаг для пропуска блоков
    size_t step = gridDim.x * blockDim.x;

    // Местные переменные для накопления минимального значения и его индекса
    int my_min = INT_MAX;
    size_t my_signal = -1;

    // Пробегаем по массиву с шагающим смещением
    for (size_t pos = gid; pos < array_length; pos += step)
    {
        int current_val = dev_max[pos];
        if (current_val < my_min)
        {
            my_min = current_val;
            my_signal = pos;
        }
    }

    // Приведение общих значений к минимуму в рамках блока
    temp[tid] = my_min;
    __syncthreads();

    // Алгоритм parallel reduction для нахождения минимального значения
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            int other_val = temp[tid + stride];
            if (other_val < temp[tid])
            {
                temp[tid] = other_val;
                my_signal = tid + stride;
            }
        }
        __syncthreads();
    }

    // Главная нить блока пишет результат в результирующий массив
    if (tid == 0)
    {
        atomicMin(result_signal, my_signal);  // Записываем минимальный индекс
    }
}

std::string intToBinaryString(size_t number, size_t n) 
{
    std::string binaryStr;
    while (number > 0) {
        binaryStr.insert(binaryStr.begin(), (number % 2) + '0');
        number /= 2;
    }
    if (binaryStr.empty()) binaryStr = "0";


    while (binaryStr.length() < n) {
        binaryStr.insert(binaryStr.begin(), '0');
    }

    return binaryStr;
}

std::string invertBinaryString(const std::string& binaryStr) 
{
    std::string invertedStr;
    for (char ch : binaryStr) {
        invertedStr.push_back(ch == '0' ? '1' : '0');
    }
    return invertedStr;
}

/*
int main()
{
    /*
    size_t n = 33; 
    size_t N = 1048576 / 8;  
    int maxD = 10000000;
    size_t* dev_bestSignals;
    int* dev_max;
    size_t* bestSignals = new size_t[N];
    int* maxs = new int[N];
    size_t bestSignal = 0;
    cudaMalloc((void**)&dev_max, N * sizeof(int));
    cudaMalloc((void**)&dev_bestSignals, N * sizeof(size_t));

    // Добавляем инструменты для замеров времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Записываем старт
    cudaEventRecord(start, 0);


        dim3 threadsPerBlock = dim3(512);
        dim3 blocksPerGrid = dim3(N / threadsPerBlock.x);
        akf_kernel << <blocksPerGrid, threadsPerBlock >> > (dev_max, dev_bestSignals, n, N);

        cudaMemcpy(bestSignals, dev_bestSignals, N * sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(maxs, dev_max, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < N; i++)
        {

            if (maxs[i] < maxD)
            {
                maxD = maxs[i];
                bestSignal = bestSignals[i];
            }
        }





    cudaFree(dev_bestSignals);
    cudaFree(dev_max);
    delete[] bestSignals;
    delete[] maxs;
    
    //invertBinaryString
    std::cout << "Best: " << invertBinaryString(intToBinaryString(bestSignal,n)) << std::endl;

    // Фиксируем конец вычисления
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("GPU execution time: %.3f s\n", elapsed_time_ms/1000);

    // Очистка событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);



    return 0;
   
    size_t n = 22;               // Длина сигнала
    size_t total_signals = pow(2, n); // Всего сигналов
    size_t grid = 12000;
    int threads = n;
    size_t block = grid / threads;
    size_t N = total_signals / block;

    dim3 blocksPerGrid = dim3(block);
    dim3 threadsPerBlock = dim3(512);
    dim3 blocksPerGrid_2 = dim3(grid / 512 + (grid % 512 != 0));

    // Переменные для запоминания текущих лучших результатов
    int current_min = INT_MAX;
    size_t current_signal = -1;

    // Память на GPU
    int* dev_max;
    size_t* dev_bestSignal;
    cudaMalloc((void**)&dev_max, block * sizeof(int));
    cudaMalloc((void**)&dev_bestSignal, sizeof(size_t));


    for (size_t iter = 1; iter < N; iter++)
    {

        max_akf_kernel << <blocksPerGrid, n>> > (dev_max, n, iter);
        /*
        find_minimum_element << <blocksPerGrid_2, threadsPerBlock >> > (dev_max, dev_bestSignal, block);

        cudaMemcpy(&min_value, dev_max, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&index, dev_bestSignal , sizeof(size_t), cudaMemcpyDeviceToHost);
        

        int* cpu_max = new int[block];  // Выделяем временное хранилище на хосте
        cudaMemcpy(cpu_max, dev_max, block * sizeof(int), cudaMemcpyDeviceToHost);

        // Линейный поиск минимума на CPU
        int min_value = 100000000000;
        size_t index = -1;
        for (size_t i = 0; i < block; i++)
        {
            if (cpu_max[i] < min_value)
            {
                min_value = cpu_max[i];
                index = i;
            }
        }

        // Освобождаем память, занятую временной переменной
        delete[] cpu_max;

        // Обновляем лучший результат
        if (min_value < current_min || (min_value == current_min && index < current_signal))
        {
            current_min = min_value;
            current_signal = index;
        }
    }


    std::cout << "Min: " << current_min << std::endl;
    std::cout << "Signal: " << invertBinaryString(intToBinaryString(current_signal,n)) << std::endl;

    // Освобождаем память
    cudaFree(dev_max);
    cudaFree(dev_bestSignal);

    return 0;
}
*/
/*
    int* akf = new int[n];
        for (int i = 0; i < n; i++) {
            akf[i] = 0;
            for (int j = 0; j < n; j++) {
                if (i + j < n) {
                    akf[i] += ((bestSignal >> (i + j) & 1) ? 1 : -1) * ((bestSignal >> (j) & 1) ? 1 : -1);
                }
            }
        }
        int max1_val = -10000000;
        int max2_val = -10000000;

        for (int i = 0; i < n; ++i) {
            if (abs(akf[i]) > max1_val) {
                max2_val = max1_val;
                max1_val = abs(akf[i]);
            }
            else if (abs(akf[i]) > max2_val && abs(akf[i]) != max1_val) {
                max2_val = abs(akf[i]);
            }
        }
        std::cout << "MAX: " << max2_val << std::endl;
        delete[] akf;
*/

/*
using namespace std;

__global__ void optimized_akf_kernel_with_generation(int offset, int* dev_max, size_t n)
{
    const size_t idx = blockIdx.x;           // Индекс блока (номер сигнала)
    const size_t tid = threadIdx.x;          // Номер потока внутри блока

    extern __shared__ int shared_mem[];      // Разделяемая память для промежуточных результатов

    // Генерируем сигнал непосредственно на GPU
    size_t unique_signal_idx = idx + offset;
    size_t signal = unique_signal_idx;       // Уникальный сигнал

    // Расчёт корреляционного значения для каждой позиции АКФ
    int akf_value = 0;
    for (size_t j = 0; j < n; j++) {
        if (tid + j < n) {                   // Проверка границы
            bool bit_i = (signal >> (tid + j)) & 1;
            bool bit_j = (signal >> j) & 1;
            akf_value += (bit_i ^ bit_j) ? -1 : 1; // XOR для проверки равенства битов
        }
    }

    // Сохраняем результат в разделяемую память
    shared_mem[tid] = akf_value;
    __syncthreads();

    // Поиск максимума второго порядка среди всех позиций АКФ
    int max1_val = INT_MIN;
    int max2_val = INT_MIN;

    for (size_t i = 0; i < n; ++i) {
        if (abs(shared_mem[i]) > max1_val) {
            max2_val = max1_val;
            max1_val = abs(shared_mem[i]);
        }
        else if (abs(shared_mem[i]) > max2_val && abs(shared_mem[i]) != max1_val) {
            max2_val = abs(shared_mem[i]);
        }
    }

    // Результат записывается обратно в глобальную память
    dev_max[idx] = max2_val;
}

int main()
{
    size_t n = 29;
    size_t N = 1048576;                     // Пакет сигналов
    size_t NM = (1 << n) / N;                // Число пакетов
    int maxD = 10000000;
    size_t bestSignal = 0;

    int* dev_max;
    cudaMalloc((void**)&dev_max, N * sizeof(int)); // Память для результирующих данных

    // Профилировка времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (size_t k = 0; k < NM; k++)
    {
        // Запуск ядра с прямым созданием сигналов на GPU
        dim3 threadsPerBlock(n);              // Потоки = длина сигнала
        dim3 blocksPerGrid(N);                // Блока на каждый сигнал пакета

        optimized_akf_kernel_with_generation << <blocksPerGrid, threadsPerBlock, n * sizeof(int) >> > (k * N, dev_max, n);

        // Получаем результаты обратно на хост
        int* maxs = new int[N];               // Временный буфер
        cudaMemcpy(maxs, dev_max, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < N; i++)
        {
            if (maxs[i] < maxD)
            {
                maxD = maxs[i];
                bestSignal = i + k * N;
            }
        }
        delete[] maxs;
    }

    // Завершаем замер времени
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("GPU execution time: %.3f seconds\n", elapsed_time_ms / 1000);

    // Финальный вывод
    cout << "Best signal: " << invertBinaryString(intToBinaryString(bestSignal, n)) << endl;

    cudaFree(dev_max);
    return 0;
}
*/


