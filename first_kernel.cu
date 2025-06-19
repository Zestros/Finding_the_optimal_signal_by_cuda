#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cmath>
#include <fstream>
#include <string>


__global__ void akf_kernel(size_t* dev_signal, int* dev_max, size_t n)
{

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int* akf = new int[n];
    for (size_t i = 0; i < n; i++) {
        akf[i] = 0;
        for (size_t j = 0; j < n; j++) {
            if (i + j < n) {
                akf[i] += ((dev_signal[idx] >> (i + j) & 1) ? 1 : -1) * ((dev_signal[idx] >> (j) & 1) ? 1 : -1);
            }
        }
    }
    int max1_val = -10000000;
    int max2_val = -10000000;

    for (size_t i = 0; i < n; ++i) {
        if (abs(akf[i]) > max1_val) {
            max2_val = max1_val;
            max1_val = abs(akf[i]);
        }
        else if (abs(akf[i]) > max2_val && abs(akf[i]) != max1_val) {
            max2_val = abs(akf[i]);
        }
    }
    dev_max[idx] = max2_val;
    delete[] akf;
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

int main()
{
    size_t n = 26;
    size_t N = 1048576 / 8;
    size_t NM = (1 << n) / N;
    int maxD = 10000000;
    size_t bestSignal = 0;
    int* dev_max;
    size_t* dev_signal;
    size_t* signals = new size_t[N];
    int* maxs = new int[N];
    cudaMalloc((void**)&dev_signal, N * sizeof(size_t));
    cudaMalloc((void**)&dev_max, N * sizeof(int));

    // Добавляем инструменты для замеров времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Записываем старт
    cudaEventRecord(start, 0);

    for (size_t k = 0; k < NM; k++)
    {

        for (size_t i = 0; i < N; i++)
        {
            signals[i] = i + k * N;
        }
        cudaMemcpy(dev_signal, signals, N * sizeof(size_t), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock = dim3(1024);
        dim3 blocksPerGrid = dim3(N / threadsPerBlock.x);
        akf_kernel << <blocksPerGrid, threadsPerBlock >> > (dev_signal, dev_max, n);

        // cudaMemcpy(signals, dev_signal, N * sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(maxs, dev_max, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < N; i++)
        {

            if (maxs[i] < maxD)
            {
                maxD = maxs[i];
                bestSignal = i + k * N;
            }
        }
    }





    cudaFree(dev_signal);
    cudaFree(dev_max);
    delete[] signals;
    delete[] maxs;


    std::cout << "Best: " << invertBinaryString(intToBinaryString(bestSignal, n)) << std::endl;

    // Фиксируем конец вычисления
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("GPU execution time: %.3f s\n", elapsed_time_ms/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

