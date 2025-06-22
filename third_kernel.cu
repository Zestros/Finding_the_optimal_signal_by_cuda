
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cmath>
#include <fstream>
#include <string>


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

