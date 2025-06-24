
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
std::string intToBinaryString(size_t number, size_t n)
{
    string binaryStr;
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
    string invertedStr;
    for (char ch : binaryStr) {
        invertedStr.push_back(ch == '0' ? '1' : '0');
    }
    return invertedStr;
}


__global__ void akf_kernel(int offset, int* dev_max, size_t n)
{
    const size_t idx = blockIdx.x;        
    const size_t tid = threadIdx.x;         

    extern __shared__ int shared_mem[];     

    size_t unique_signal_idx = idx + offset;
    size_t signal = unique_signal_idx;     

    int akf_value = 0;
    for (size_t j = 0; j < n; j++) {
        if (tid + j < n) {                 
            bool bit_i = (signal >> (tid + j)) & 1;
            bool bit_j = (signal >> j) & 1;
            akf_value += (bit_i ^ bit_j) ? -1 : 1;
        }
    }

    shared_mem[tid] = akf_value;          
    __syncthreads();                      

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

    dev_max[idx] = max2_val;               
}

int main()
{
    ofstream results_file("results_but_besty.txt"); 

   
    const size_t MAX_N = 33;

    
    for (size_t n = 5; n <= MAX_N; n++)
    {
        size_t N = 1048576;  
        size_t p = pow(2, n);
        if (p < N)
        {
            N = p;
        }
        size_t one = 1;
        size_t NM = (one << n) / N;     
        if (NM == 0) NM = 1;
        int maxD = 10000000;                
        size_t bestSignal = 0;               

        int* dev_max;                        
        cudaMalloc((void**)&dev_max, N * sizeof(int)); 

        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);           

        
        for (size_t k = 0; k < NM; k++)
        {
            dim3 threadsPerBlock(n);            
            dim3 blocksPerGrid(N);          

           
            akf_kernel << <blocksPerGrid, threadsPerBlock, n * sizeof(int) >> > (k * N, dev_max, n);

            
            int* maxs = new int[N];
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

        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        
        float elapsed_time_ms;
        cudaEventElapsedTime(&elapsed_time_ms, start, stop);

        
        cudaFree(dev_max);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        
        string best_binary = invertBinaryString(intToBinaryString(bestSignal, n));

        
        results_file  << best_binary << " " << n << " " << maxD << " " << elapsed_time_ms << '\n';
        cout << best_binary << " " << n << " " << maxD << " " << elapsed_time_ms << '\n';
    }

    
    results_file.close();

    return 0;
}
