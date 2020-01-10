#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#define VECTOR_SIZE 4096
#define VECTOR_NUM 4096
#define blocksize 8
#define TILE_WIDTH 16

using namespace std;

__global__ void rbf_cuda(const float *a, const float *b, float *c, int vec_size, int vec_num)
// use cuda to calculate rbf with only global memory
{
    // get the thread parameters
    int index_a = blockIdx.y * blockDim.y + threadIdx.y;
    int index_b = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp = 0;
    //if (index_a >= VECTOR_NUM || index_b >= VECTOR_NUM) return;
    for (int i = 0; i < vec_size; ++i)
    {
        tmp += (a[index_a * vec_size + i] - b[index_b * vec_size + i]) * \
               (a[index_a * vec_size + i] - b[index_b * vec_size + i]);
    }
    tmp = sqrt(tmp);
    c[index_a * vec_num + index_b] = tmp;
}

__global__ void rbf_cuda_shareMem(const float *a, const float *b, float *c, int vec_size, int vec_num)
// use cuda to calculate rbf with global memory and shared memory
{
    // shared memory allocation
    __shared__ float a_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_s[TILE_WIDTH][TILE_WIDTH];

    // thread parameters
    int index_a = blockIdx.y * blockDim.y + threadIdx.y;
    int index_b = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float tmp = 0;
    for (int i = 0; i < vec_size / TILE_WIDTH; i++)
    {
        // in each tiling, a thread load two data into shared memory
        // which will be used for the whole thread block.
        a_s[ty][tx] = \
            a[index_a * vec_size + i * TILE_WIDTH + tx];
        b_s[ty][tx] = \
            b[index_b * vec_size + i * TILE_WIDTH + tx];
        // synchronize threads to avoid data race
        __syncthreads();
        
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            tmp += (a_s[ty][j] - b_s[ty][j]) * \
                   (a_s[ty][j] - b_s[ty][j]);
        }
        __syncthreads();
    }
    
    c[index_a * vec_num + index_b] = sqrt(tmp);
}

void rbf_cpu(float *a, float *b, float *c)
// naive CPU function
{
    float tmp = 0;
    for (int i = 0; i < VECTOR_NUM; i++)
        for (int j = 0; j < VECTOR_NUM; j++)
        {
            tmp = 0;
            for (int k = 0; k < VECTOR_SIZE; k++)
            {
                tmp += (a[i * VECTOR_SIZE + k] - b[j * VECTOR_SIZE + k]) * \
                       (a[i * VECTOR_SIZE + k] - b[j * VECTOR_SIZE + k]);
            }
            c[i * VECTOR_NUM + j] = sqrt(tmp);
        }
}

void genNum(float *m, int num, int size, float value)
{
    for (int i = 0; i < num; ++i)
        for (int j = 0; j < size; ++j)
            m[i * size + j] = value;
}

int main(int argc, char *argv[])
{
    // parse command line
    int mode = 0;
    if (argc == 1)
    {
        cout << "Too few arguments, please add \'-c\', \'-n\' or \'-s\'.\n";
        return 0;
    }

    if (!strcmp(argv[1], "-c")) mode = 0;
    else if (!strcmp(argv[1], "-n")) mode = 1;
    else if (!strcmp(argv[1], "-s")) mode = 2;
    else
    {
        cout << "Invalid run mode!\n";
        return 0;
    }

    clock_t start, end;
    
    // memory on host(CPU)
    float *a, *b, *c;
    int calMatrixSize = sizeof(float) * VECTOR_SIZE * VECTOR_NUM;
    int ansMatrixSize = sizeof(float) * VECTOR_NUM * VECTOR_NUM;
   
    a = (float*)malloc(calMatrixSize);
    b = (float*)malloc(calMatrixSize);
    c = (float*)malloc(ansMatrixSize);
    
    // initialize the memory of matrix on CPU
    genNum(a, VECTOR_NUM, VECTOR_SIZE, 1.0);
    genNum(b, VECTOR_NUM, VECTOR_SIZE, 2.0);
    genNum(c, VECTOR_NUM, VECTOR_NUM, 3.0); // only for debug, can be deleted later

    // memory on device(GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, calMatrixSize);
    cudaMalloc((void**)&d_b, calMatrixSize);
    cudaMalloc((void**)&d_c, ansMatrixSize);
    
    // cpu mode
    if (mode == 0)
    {
        cout << "Using CPU to calculate...\n";
        // start timing
        start = clock();
        rbf_cpu(a, b, c);
        end = clock();
    }

    // normal mode
    else if (mode == 1)
    {
        cout << "Using GPU with global memory to calculate...\n";
        // prepare the size of block and grid
        dim3 blockSize(blocksize, blocksize);
        dim3 gridSize((VECTOR_NUM + blocksize - 1) / blocksize, (VECTOR_NUM + blocksize - 1) / blocksize);
        
        // start timing
        start = clock();

        // copy the matrix from host to device
        cudaMemcpy((void*)d_a, (void*)a, calMatrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_b, (void*)b, calMatrixSize, cudaMemcpyHostToDevice);
        
        // calculate RBF kernel function
        rbf_cuda<<<gridSize, blockSize>>>(d_a, d_b, d_c, VECTOR_SIZE, VECTOR_NUM);
        
        // copy the ans matrix from device to host
        cudaMemcpy((void*)c, (void*)d_c, ansMatrixSize, cudaMemcpyDeviceToHost);
        end = clock();
    }

    // shared memory mode
    else
    {
        cout << "Using GPU with global and shared memory to calculate...\n";
        // prepare the size of block and grid
        dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
        dim3 gridSize((VECTOR_NUM + TILE_WIDTH - 1) / TILE_WIDTH, (VECTOR_NUM + TILE_WIDTH - 1) / TILE_WIDTH);
        
        // start timing
        start = clock();

        // copy the matrix from host to device
        cudaMemcpy((void*)d_a, (void*)a, calMatrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_b, (void*)b, calMatrixSize, cudaMemcpyHostToDevice);
        
        // calculate RBF kernel function
        rbf_cuda_shareMem<<<gridSize, blockSize>>>(d_a, d_b, d_c, VECTOR_SIZE, VECTOR_NUM);
        
        // copy the ans matrix from device to host
        cudaMemcpy((void*)c, (void*)d_c, ansMatrixSize, cudaMemcpyDeviceToHost);
        end = clock();
    }

    cout << "Time consuming: " << double(end - start) / CLOCKS_PER_SEC << "s\n";
    
    // check the error
    float max_error = 0.0;
    float correct_ans = sqrt(VECTOR_SIZE);
    int error_cnt = 0;
    for (int i = 0; i < VECTOR_NUM; ++i)
        for (int j = 0; j < VECTOR_NUM; ++j)
        {
            float error = c[i * VECTOR_NUM + j] - correct_ans;
            error = error >= 0 ? error : -1 * error;
            if (error != 0.0) error_cnt++;
            if (error > max_error) max_error = error;
        }

    cout << "Max error: " << max_error << endl;
    cout << "Total error counts: " << error_cnt << endl;
    
    // free all memory 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    return 0;
}
