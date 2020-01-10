#ifndef _GEMV_CU_
#define _GEMV_CU_

namespace gemv {
  // shared memory, tile 128: 4.4 iter/s
  // constant memory, thread 16: 4.96 iter/s
  int BLOCK_SIZE_gemv = 32;
  const int MAX_gemv = 12544;
  cudaError_t cudaStatus;
  __constant__ float vector_gemv[MAX_gemv];
  // gemv
  // gemv_kerkel
  template <typename Dtype>
  __global__ void kernel_gemv_n(const int M, const int N, const Dtype alpha, 
      const Dtype *a, const Dtype *x, const Dtype beta, Dtype *y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    Dtype ans = 0;
    /**
    * Share Memory and Tiling Method
    * int ty = threadIdx.y;
    * __shared__ Dtype x_tile[BLOCK_SIZE_gemv];
    * if (row < M)
    * {
    *   for (int i = 0; i < N; i += BLOCK_SIZE_gemv)
    *   {
    *     x_tile[ty] = (i + ty < N) ? x[i + ty] : 0;
    *     __syncthreads();
    * 
    *     for (int j = 0; j < BLOCK_SIZE_gemv && i + j < N; ++j)
    *       ans += a[row * N + (i + j)] * x_tile[j];
    *     __syncthreads();
    *   }
    *   ans *= alpha;
    *   if (beta == 0)
    *     y[row] = ans;
    *   else
    *     y[row] = ans + beta * y[row];
    * }
    */
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
  __global__ void kernel_gemv_t(const int M, const int N, const Dtype alpha, 
      const Dtype *a, const Dtype *x, const Dtype beta, Dtype *y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    Dtype ans = 0;
    /**
    * Share Memory and Tiling Method
    * int ty = threadIdx.y;
    * __shared__ Dtype x_tile[BLOCK_SIZE_gemv];
    * if (row < M)
    * {
    *   for (int i = 0; i < N; i += BLOCK_SIZE_gemv)
    *   {
    *     x_tile[ty] = (i + ty < N) ? x[i + ty] : 0;
    *     __syncthreads();
    * 
    *     for (int j = 0; j < BLOCK_SIZE_gemv && i + j < N; ++j)
    *       ans += a[row + (i + j) * M] * x_tile[j];
    *     __syncthreads();
    *   }
    *   ans *= alpha;
    *   if (beta == 0)
    *     y[row] = ans;
    *   else
    *     y[row] = ans + beta * y[row];
    * }
    */
    if (row < M)
    {
      for (int i = 0; i < N; ++i)
        ans += a[row + i * M] * x[i];
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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    Dtype ans = 0;
    if (row < M)
    {
      for (int i = 0; i < N; ++i)
        ans += a[row * N + i] * vector_gemv[i];
      ans *= alpha;
      if (beta == 0)
        y[row] = ans;
      else
        y[row] = ans + beta * y[row];
    }
  }

  template <typename Dtype>
  __global__ void kernel_gemv_t_const(const int M, const int N, const Dtype alpha, 
      const Dtype *a, const Dtype beta, Dtype *y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    Dtype ans = 0;
    if (row < M)
    {
      for (int i = 0; i < N; ++i)
        ans += a[row + i * M] * vector_gemv[i];
      ans *= alpha;
      if (beta == 0)
        y[row] = ans;
      else
        y[row] = ans + beta * y[row];
    }
  }

  void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
      const int N, const float alpha, const float* A, const float* x,
      const float beta, float* y) {
    if (TransA == CblasNoTrans)
    {
      // if (M < 32)
        BLOCK_SIZE_gemv = 16;
      // else
        // BLOCK_SIZE_gemv = 32;
      int grid_rows = (M + BLOCK_SIZE_gemv - 1) / BLOCK_SIZE_gemv;
      dim3 gridSize(1, grid_rows);
      dim3 blockSize(1, BLOCK_SIZE_gemv);
      cudaMemcpyToSymbol(vector_gemv, x, N * sizeof(float), 0, cudaMemcpyDeviceToDevice);
      kernel_gemv_n_const<float><<<gridSize, blockSize>>>(M, N, alpha, A, beta, y);
    }
    else
    {
      // if (N < 32)
        BLOCK_SIZE_gemv = 16;
      // else
        // BLOCK_SIZE_gemv = 32;
      int grid_rows = (N + BLOCK_SIZE_gemv - 1) / BLOCK_SIZE_gemv;
      dim3 gridSize(1, grid_rows);
      dim3 blockSize(1, BLOCK_SIZE_gemv);
      cudaMemcpyToSymbol(vector_gemv, x, M * sizeof(float), 0, cudaMemcpyDeviceToDevice);
      kernel_gemv_t_const<float><<<gridSize, blockSize>>>(N, M, alpha, A, beta, y);
    }
  }

} // namespace gemv
#endif // _GEMV_CU_