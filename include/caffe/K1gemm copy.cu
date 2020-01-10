#ifndef _K1GEMM_CU_
#define _K1GEMM_CU_

namespace K1gemm {

    #define BLOCK_SIZE_K1 16
    const int MAX_M = 2048;
    const int MAX_N = 12544;

    __constant__ float cA[MAX_M], cB[MAX_N];
    
    __global__ void kernel_gemm(const int M, const int N, 
        const float alpha, const float *a, const float *b, const float beta,
        float *c) {
        // __shared__ float tile_a[BLOCK_SIZE_K1];
        // __shared__ float tile_b[BLOCK_SIZE_K1];
    
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        int row = blockIdx.y * BLOCK_SIZE_K1 + ty;
        int col = blockIdx.x * BLOCK_SIZE_K1 + tx;

        // if (row < M && tx == 0)
        //     tile_a[ty] = a[row];
        // if (col < N && ty == 0)
        //     tile_b[tx] = b[col];
        // __syncthreads();
        if (row < M && col < N) {

            // float ans = tile_a[ty] * tile_b[tx];
            // __syncthreads();
            
            float ans = cA[row] * cB[col];
            ans *= alpha;

            if (beta != 0)
                c[row * N + col] = c[row * N + col] * beta + ans;
            else
                c[row * N + col] = ans;
        }
    }
    
    // gemm kernel
    // 4: 1.3
    // 7: 2.7
    // 8: 4
    // 14: 4.15
    // 16: 5.08, pragma unroll(4): 5.18, manually unroll: 5.25
    // 18: 4.5
    // 20: 4.5
    // 32: 4.85
    // cublas: 10.5
    // gemm interface
    void caffe_gpu_gemm(const int M, const int N, 
        const float alpha, const float* A, const float* B, const float beta,
        float* C) {
        int grid_rows = (M + BLOCK_SIZE_K1 - 1) / BLOCK_SIZE_K1;
        int grid_cols = (N + BLOCK_SIZE_K1 - 1) / BLOCK_SIZE_K1;
        dim3 gridSize(grid_cols, grid_rows);
        dim3 blockSize(BLOCK_SIZE_K1, BLOCK_SIZE_K1);

        int A_size = M * sizeof(float);
        int B_size = N * sizeof(float);
        // __constant__ float cA[A_size], cB[B_size];
        cudaMemcpyToSymbol(cA, A, A_size);
        cudaMemcpyToSymbol(cB, B, B_size);

        kernel_gemm<<<gridSize, blockSize>>>(M, N, alpha, A, B, beta, C);
        
        // cudaError_t er1 = cudaPeekAtLastError();
        // CUDA_CHECK(er1);
    }
    
} // K1gemm

#endif // _K1GEMM_CU_