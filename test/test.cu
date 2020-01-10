#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include "cublas_v2.h"
#define LEN 2000

using namespace std;

template <typename Dtype>
__global__ void kernel_dot_2000(const int N, const Dtype *X, const Dtype *Y, 
    Dtype *d_out) {
  int idx = threadIdx.x;
  __shared__ Dtype buffer[256];
  if (idx >= 250)
      buffer[idx] = 0;
  else
  {
    Dtype localAns = 0;
    for (int i = idx; i < 2000; i += 250)
      localAns += X[i] * Y[i];
    buffer[idx] = localAns;
  }
  __syncthreads();

  for (int i = 2; i <= 256; i *= 2)
  {
    if (idx % i == 0)
      buffer[idx] += buffer[idx + i / 2];
    __syncthreads();
  }

  *d_out = buffer[0];
}

template <typename Dtype>
__global__ void kernel_dot_n(const int N, const Dtype *X, const Dtype *Y, 
    Dtype *d_out) {
  int idx = threadIdx.x;
  __shared__ Dtype buffer[256];

  Dtype localAns = 0;
  for (int i = idx; i < N - 256; i += 256)
    localAns += X[i] * Y[i];
  localAns += X[N - 256 + idx] * Y[N - 256 + idx];
  buffer[idx] = localAns;
  
  __syncthreads();

  for (int i = 2; i <= 256; i *= 2)
  {
    if (idx % i == 0)
      buffer[idx] += buffer[idx + i / 2];
    __syncthreads();
  }

  *d_out = buffer[0];
}

int main()
{
    float *x, *y, *a;
    cudaError_t cudaStatus;
    int size = 2 * LEN * sizeof(float);

    a = (float*)malloc(2 * sizeof(float));
    x = (float*)malloc(size);
    y = (float*)malloc(size);

    for (int i = 0; i < LEN; ++i)
    {
      x[i] = 1;
      y[i] = 1;
    }
    for (int i = LEN; i < 2 * LEN; ++i)
    {
      x[i] = 2;
      y[i] = 2;
    }
    
    float *d_a, *d_x, *d_y;
    cudaMalloc((void**)&d_a, 2 * sizeof(float));
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);

    cudaMemcpy((void*)d_x, (void*)x, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, size, cudaMemcpyHostToDevice);

    cudaStream_t stream[2];
    for(int i = 0; i < 2; i++)
    {
      cudaStreamCreate(&stream[i]);
    }

    kernel_dot_2000<float><<<1, 256, 0, stream[0]>>>(LEN, d_x, d_y, d_a);
    kernel_dot_n<float><<<1, 256, 0, stream[1]>>>(1500, d_x + 2000, d_y + 2000, d_a + 1);
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
      cout << "error!!!" << endl;
      goto Error;
    }

    cudaMemcpy((void*)a, (void*)d_a, sizeof(float) * 2, cudaMemcpyDeviceToHost);

    cout << *a << endl;
    cout << *(a + 1) << endl;

Error:
    for(int i = 0; i < 2; i++)
    {
      cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
    free(a);
    free(x);
    free(y);

    return 0;
}