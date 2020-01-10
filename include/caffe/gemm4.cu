#ifndef _GEMM4_CU_
#define _GEMM4_CU_

namespace gemm4 {

#define BLOCK_SIZE_4 4
#define taxtb_4 (tile_a[ty][0]*tile_b[0][tx]+tile_a[ty][1]*tile_b[1][tx]+tile_a[ty][2]*tile_b[2][tx]+tile_a[ty][3]*tile_b[3][tx])

template <typename Dtype>
__global__ void kernel_gemm_nn(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE_4][BLOCK_SIZE_4];
  __shared__ Dtype tile_b[BLOCK_SIZE_4][BLOCK_SIZE_4];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_4 + ty;
  int col = blockIdx.x * BLOCK_SIZE_4 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE_4; i += BLOCK_SIZE_4)
  {
    tile_a[ty][tx] = (row < M) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = (col < N) ? b[(i + ty) * N + col] : 0;
    __syncthreads();
    ans += taxtb_4;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += taxtb_4;
  __syncthreads();  

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
  __shared__ Dtype tile_a[BLOCK_SIZE_4][BLOCK_SIZE_4];
  __shared__ Dtype tile_b[BLOCK_SIZE_4][BLOCK_SIZE_4];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_4 + ty;
  int col = blockIdx.x * BLOCK_SIZE_4 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE_4; i += BLOCK_SIZE_4)
  {
    tile_a[ty][tx] = (row < M) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = (col < N) ? b[(i + ty) * N + col] : 0;
    __syncthreads();
    ans += taxtb_4;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += taxtb_4;
  __syncthreads();

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
  __shared__ Dtype tile_a[BLOCK_SIZE_4][BLOCK_SIZE_4];
  __shared__ Dtype tile_b[BLOCK_SIZE_4][BLOCK_SIZE_4];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_4 + ty;
  int col = blockIdx.x * BLOCK_SIZE_4 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE_4; i += BLOCK_SIZE_4)
  {
    tile_a[ty][tx] = (row < M) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = (col < N) ? b[col * K + i + ty] : 0;
    __syncthreads();
    ans += taxtb_4;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0;
  __syncthreads();
  ans += taxtb_4;
  __syncthreads();

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
  __shared__ Dtype tile_a[BLOCK_SIZE_4][BLOCK_SIZE_4];
  __shared__ Dtype tile_b[BLOCK_SIZE_4][BLOCK_SIZE_4];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_4 + ty;
  int col = blockIdx.x * BLOCK_SIZE_4 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE_4; i += BLOCK_SIZE_4)
  {
    tile_a[ty][tx] = (row < M) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = (col < N) ? b[col * K + i + ty] : 0; //
    __syncthreads(); 
    ans += taxtb_4;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0; //
  __syncthreads();
  ans += taxtb_4;
  __syncthreads();

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
  int grid_rows = (M + BLOCK_SIZE_4 - 1) / BLOCK_SIZE_4;
  int grid_cols = (N + BLOCK_SIZE_4 - 1) / BLOCK_SIZE_4;
  dim3 gridSize(grid_cols, grid_rows);
  dim3 blockSize(BLOCK_SIZE_4, BLOCK_SIZE_4);
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