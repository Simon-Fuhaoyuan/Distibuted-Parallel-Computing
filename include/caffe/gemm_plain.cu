#ifndef _GEMM_PLAIN_CU_
#define _GEMM_PLAIN_CU_

namespace gemm_plain {

#define BLOCK_SIZE_PLAIN 16

template <typename Dtype>
__global__ void kernel_gemm_nn(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  Dtype ans = 0;

  for (int i = 0; i < K; i++) {
      ans += a[row * K + i] * b[i * N + col];
  }

  ans *= alpha;
  if (row < M && col < N)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}
template <typename Dtype>
__global__ void kernel_gemm_tn(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  Dtype ans = 0;

  for (int i = 0; i < K; i++) {
      ans += a[i * K + row] * b[i * N + col];
  }

  ans *= alpha;
  if (row < M && col < N)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}
template <typename Dtype>
__global__ void kernel_gemm_nt(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  Dtype ans = 0;

  for (int i = 0; i < K; i++) {
      ans += a[row * K + i] * b[col * N + i];
  }

  ans *= alpha;
  if (row < M && col < N)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}
template <typename Dtype>
__global__ void kernel_gemm_tt(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x + tx;
  Dtype ans = 0;

  for (int i = 0; i < K; i++) {
      ans += a[i * K + row] * b[col * N + i];
  }

  ans *= alpha;
  if (row < M && col < N)
  {
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
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int grid_rows = (M + BLOCK_SIZE_PLAIN - 1) / BLOCK_SIZE_PLAIN;
  int grid_cols = (N + BLOCK_SIZE_PLAIN - 1) / BLOCK_SIZE_PLAIN;
  dim3 gridSize(grid_cols, grid_rows);
  dim3 blockSize(BLOCK_SIZE_PLAIN, BLOCK_SIZE_PLAIN);
  if (TransA == CblasNoTrans && TransB == CblasNoTrans)
    kernel_gemm_nn<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
  else if (TransA != CblasNoTrans && TransB == CblasNoTrans)
    kernel_gemm_tn<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
  else if (TransA == CblasNoTrans && TransB != CblasNoTrans)
    kernel_gemm_nt<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
  else
    kernel_gemm_tt<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);

  // cudaError_t er1 = cudaPeekAtLastError();
  // CUDA_CHECK(er1);
}

}
#endif