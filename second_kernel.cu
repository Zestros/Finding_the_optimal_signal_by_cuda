
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cmath>
#include <fstream>
#include <string>


__global__ void akf_kernel(int* dev_max, size_t* dev_bestSignals, size_t n, size_t N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int* akf = new int[n];
    size_t loc_bestSignal = 0;
    int pr_dev_max = 100000000;
    size_t mxidx = 1;
    mxidx <<= n;
    int max1_val, max2_val;
    size_t k, ind,i, j;
    for (k = idx, ind = 0; ind < mxidx / N; k += N, ind++)
    {
        for (i = 0; i < n; i++) {
            akf[i] = 0;
            for (j = 0; j < n; j++) {
                if (i + j < n) {
                    akf[i] += ((k >> (i + j) & 1) ? 1 : -1) * ((k >> (j) & 1) ? 1 : -1);
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
    size_t n = 28;
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
        for (size_t i = 0; i < N; i++) 
        {
            if (maxs[i] < maxD || (maxs[i] == maxD && bestSignals[i] < bestSignal)) 
            {
                maxD = maxs[i];     
                bestSignal = bestSignals[i]; 
            }
        }

    }





    cudaFree(dev_bestSignals);
    cudaFree(dev_max);
    delete[] bestSignals;
    delete[] maxs;

    //invertBinaryString
    std::cout << "Best: " << invertBinaryString(intToBinaryString(bestSignal, n)) << std::endl;

    // Фиксируем конец вычисления
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    printf("GPU execution time: %.3f s\n", elapsed_time_ms / 1000);

    // Очистка событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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



    return 0;
}




