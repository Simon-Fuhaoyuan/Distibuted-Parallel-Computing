#ifndef _GEMV_CU_
#define _GEMV_CU_

namespace gemv {

    __constant__ float vec[1024];
    const int BLOCK_SIZE_gemv = 16;
    cudaError_t cudaStatus;

    template <typename Dtype>
    __global__ void kernel_gemv_n(const int M, const int N, const Dtype alpha, 
        const Dtype *a, const Dtype *x, const Dtype beta, Dtype *y) {
        int row = blockIdx.y * BLOCK_SIZE_gemv + threadIdx.y;
        int ty = threadIdx.y;
        Dtype ans = 0;
        if (row < M)
        {
            for (int i = 0; i < N; ++i)
            ans += a[row * N + i] * x[i];
            ans *= alpha;
            if (beta == 0)
            y[row] = ans;
            else
            y[row] = ans + beta * y[row];
        }
    }

    template <typename Dtype>
    __global__ void kernel_gemv_n_const(const int M, const int N, const Dtype alpha, 
        const Dtype *a, const Dtype beta, Dtype *y) {
        int row = blockIdx.y * BLOCK_SIZE_gemv + threadIdx.y;
        Dtype ans = 0;
        if (row < M)
        {
            for (int i = 0; i < N; ++i)
            ans += a[row * N + i] * vec[i];
            ans *= alpha;
            if (beta == 0)
            y[row] = ans;
            else
            y[row] = ans + beta * y[row];
        }
    }

    void caffe_gpu_gemv(const int M,const int N, const float alpha, 
        const float* A, const float* x, const float beta, float* y) {
        int grid_rows = (M + BLOCK_SIZE_gemv - 1) / BLOCK_SIZE_gemv;
        dim3 gridSize(1, grid_rows);
        dim3 blockSize(1, BLOCK_SIZE_gemv);
        // for (int i = 0; i < N; ++i) {
        //     cudaStatus = cudaMemcpyToSymbol(*vec, x + i, sizeof(float), sizeof(float) * i, cudaMemcpyDeviceToDevice);
        //     std::cout << "Error: " << cudaStatus << std::endl;
        // }
        cudaStatus = cudaMemcpyToSymbol(vec, x, N * sizeof(float));
        std::cout << "Error: " << cudaStatus << std::endl;
        kernel_gemv_n_const<float><<<gridSize, blockSize>>>(M, N, alpha, A, beta, y);
    }
}

#endif // _GEMV_CU_