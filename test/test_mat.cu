#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include "cublas_v2.h"
#include "gemv.cu"
#define MM 4
#define NN 3

using namespace std;

int main(int argc, char *argv[])
{
    int mode = atoi(argv[1]);
    srand((unsigned)time(0));
    float *a, *x, *y;

    a = (float*)malloc(MM * NN * sizeof(float));
    x = (float*)malloc(NN * sizeof(float));
    y = (float*)malloc(MM * sizeof(float));
    float alpha = 1.0;
    float beta = 0;

    for (int i = 0; i < MM; ++i)
        for (int j = 0; j < NN; ++j)
            a[i * NN + j] = i * NN + j;
    
    for (int i = 0; i < NN; ++i)
        x[i] = i;
    
    float *d_a, *d_x, *d_y;
    cudaMalloc((void**)&d_a, MM * NN * sizeof(float));
    cudaMalloc((void**)&d_x, NN * sizeof(float));
    cudaMalloc((void**)&d_y, MM * sizeof(float));

    cudaMemcpy((void*)d_a, (void*)a, MM * NN* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_x, (void*)x, NN * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start, end;
    if (mode == 0)
    {
        start = clock();
        gemv::caffe_gpu_gemv(MM, NN, alpha, d_a, d_x, beta, d_y);
    }

    cudaMemcpy((void*)y, (void*)d_y, sizeof(float) * MM, cudaMemcpyDeviceToHost);
    for (int i = 0; i < MM; ++i)
        cout << y[i] << ' ';
    cout << endl;

Error:
    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
    free(a);
    free(x);
    free(y);

    return 0;
}