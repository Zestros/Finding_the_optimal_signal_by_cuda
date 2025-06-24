
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
    int* akf = new int[n];  // Динамическое выделение памяти на устройстве
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

    const size_t MAX_N = 31; 


    std::ofstream outputFile("results.txt");


    for (size_t n = 5; n <= MAX_N; ++n)
    {
        size_t N;
        size_t NM;
        if (n < 17)
        {
            N = pow(2,n);
            NM = 1;
        }
        else
        {
            N = 1048576 / 8;
            NM = (1 << n) / N;
        }
        

 
        int maxD = 10000000;
        size_t bestSignal = 0;

        int* dev_max;
        size_t* dev_signal;
        size_t* signals = new size_t[N];
        int* maxs = new int[N];
        cudaMalloc((void**)&dev_signal, N * sizeof(size_t));
        cudaMalloc((void**)&dev_max, N * sizeof(int));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);


        for (size_t k = 0; k < NM; k++)
        {
            for (size_t i = 0; i < N; i++)
            {
                signals[i] = i + k * N;
            }


            cudaMemcpy(dev_signal, signals, N * sizeof(size_t), cudaMemcpyHostToDevice);

            dim3 threadsPerBlock;
            dim3 blocksPerGrid; 
            if (n < 17)
            {
                threadsPerBlock = dim3(16);
                blocksPerGrid = dim3(N / 16);
            }
            else
            {
                threadsPerBlock = dim3(1024);
                blocksPerGrid = dim3(N / threadsPerBlock.x);
            }



            akf_kernel << <blocksPerGrid, threadsPerBlock >> > (dev_signal, dev_max, n);


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


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);


        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

 
        outputFile << invertBinaryString(intToBinaryString(bestSignal, n))<< " " <<n
            << " " << maxD << " " << elapsed_time_ms << "\n";


        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }


    outputFile.close();

    return 0;
}
