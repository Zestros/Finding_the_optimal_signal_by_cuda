#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <algorithm>
#include <chrono>
#define N 5
#define MIN_VAL -1e9
#define MAX_VAL 1e9
using namespace std;

int getSignalVal(int bit) {
    if (bit) {
        return 1;
    }
    else {
        return -1;
    }
}

int getBit(unsigned int n, unsigned int k) {
    return (n >> k) % 2;
}

int* akf(unsigned int signal, unsigned int n) {
    int* akf = new int[n];
    for (int i = 0; i < n; i++) {
        int k = 0;
        akf[i] = 0;
        for (int j = 0; j < n; j++) {
            if (i + j < n) {
                k++;
                akf[i] += getSignalVal(getBit(signal, i + j)) * getSignalVal(getBit(signal, j));
            }
        }
        akf[i] = abs(akf[i]);
    }
    return akf;
}

int getSecondMaxAndDeleteArray(int* arr, unsigned int n) {
    int mx1 = MIN_VAL;
    int mx2 = MIN_VAL;
    for (int i = 0; i < n; i++) {
        if (arr[i] > mx2) {
            mx2 = arr[i];
            if (arr[i] > mx1) {
                mx2 = mx1;
                mx1 = arr[i];
            }
        }
    }
    delete[] arr;
    return mx2;
}

std::string getAllBits(unsigned int signal, unsigned int n) {
    std::string res = "";
    unsigned int tmp;
    for (unsigned int i = 0; i < n; i++) {
        tmp = signal % 2;
        if (tmp) {
            res += '1';
        }
        else {
            res += '0';
        }
        signal >>= 1;
    }
    std::reverse(res.begin(), res.end());
    return res;
}

std::string getAllBitsInverse(unsigned int signal, unsigned int n) {
    std::string res = "";
    unsigned int tmp;
    for (unsigned int i = 0; i < n; i++) {
        tmp = signal % 2;
        if (tmp) {
            res += '0';
        }
        else {
            res += '1';
        }
        signal >>= 1;
    }
    std::reverse(res.begin(), res.end());
    return res;
}

void printLinearOnRange(unsigned int begin, unsigned int end) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::ofstream out("linear_output.txt");

    for (unsigned int n = begin; n < end; n++) {
        unsigned int signal = 0;
        int mn = INT_MAX;
        unsigned int mn_signal = 0;

        int tmp;
        while (signal < (1u << n)) {
            tmp = getSecondMaxAndDeleteArray(akf(signal, n), n);
            if (mn > tmp) {
                mn = tmp;
                mn_signal = signal;
            }
            signal++;
        }

        std::string bits_inverse = getAllBitsInverse(mn_signal, n);
        auto finish_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count();
        std::cout << bits_inverse << " " << n << " " << mn << " " << duration <<std::endl;
        out << bits_inverse << " " << n << " " << mn << " " << duration <<std::endl;
    }



}

int main()
{
    setlocale(LC_ALL, "Russian");
    printLinearOnRange(5, 25);

    return 0;
}

