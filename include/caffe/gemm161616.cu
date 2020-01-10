#ifndef _GEMM161616_CU_
#define _GEMM161616_CU_

namespace gemm161616 {

#define BLOCK_SIZE_161616 16 
#define tile_a_x_tile_b (tile_a[ty][0]*tile_b[0][tx]+tile_a[ty][1]*tile_b[1][tx]+tile_a[ty][2]*tile_b[2][tx]+tile_a[ty][3]*tile_b[3][tx]+ \
  tile_a[ty][4]*tile_b[4][tx]+tile_a[ty][5]*tile_b[5][tx]+tile_a[ty][6]*tile_b[6][tx]+tile_a[ty][7]*tile_b[7][tx]+ \
  tile_a[ty][8]*tile_b[8][tx]+tile_a[ty][9]*tile_b[9][tx]+tile_a[ty][10]*tile_b[10][tx]+tile_a[ty][11]*tile_b[11][tx]+ \
  tile_a[ty][12]*tile_b[12][tx]+tile_a[ty][13]*tile_b[13][tx]+tile_a[ty][14]*tile_b[14][tx]+tile_a[ty][15]*tile_b[15][tx])


template <typename Dtype>
__global__ void kernel_gemm_nn(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE_161616][BLOCK_SIZE_161616];
  __shared__ Dtype tile_b[BLOCK_SIZE_161616][BLOCK_SIZE_161616];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_161616 + ty;
  int col = blockIdx.x * BLOCK_SIZE_161616 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K ; i += BLOCK_SIZE_161616)
  {
    tile_a[ty][tx] = a[row * K + i + tx];
    tile_b[ty][tx] = b[(i + ty) * N + col];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }


  ans *= alpha;
  if (beta != 0)
    c[row * N + col] = c[row * N + col] * beta + ans;
  else
    c[row * N + col] = ans;
  
}

template <typename Dtype>
__global__ void kernel_gemm_tn(const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE_161616][BLOCK_SIZE_161616];
  __shared__ Dtype tile_b[BLOCK_SIZE_161616][BLOCK_SIZE_161616];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_161616 + ty;
  int col = blockIdx.x * BLOCK_SIZE_161616 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K; i += BLOCK_SIZE_161616)
  {
    tile_a[ty][tx] = a[(i + tx) * M + row];
    tile_b[ty][tx] = b[(i + ty) * N + col];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  ans *= alpha;
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  
}

template <typename Dtype>
__global__ void kernel_gemm_nt(const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE_161616][BLOCK_SIZE_161616];
  __shared__ Dtype tile_b[BLOCK_SIZE_161616][BLOCK_SIZE_161616];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_161616 + ty;
  int col = blockIdx.x * BLOCK_SIZE_161616 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K; i += BLOCK_SIZE_161616)
  {
    tile_a[ty][tx] = a[row * K + i + tx];
    tile_b[ty][tx] = b[col * K + i + ty];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  ans *= alpha;
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  
}

template <typename Dtype>
__global__ void kernel_gemm_tt(const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE_161616][BLOCK_SIZE_161616];
  __shared__ Dtype tile_b[BLOCK_SIZE_161616][BLOCK_SIZE_161616];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE_161616 + ty;
  int col = blockIdx.x * BLOCK_SIZE_161616 + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K; i += BLOCK_SIZE_161616)
  {
    tile_a[ty][tx] = a[(i + tx) * M + row];
    tile_b[ty][tx] = b[col * K + i + ty];
    __syncthreads(); 
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  ans *= alpha;
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  
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
  int grid_rows = M / BLOCK_SIZE_161616;
  int grid_cols = N / BLOCK_SIZE_161616;
  dim3 gridSize(grid_cols, grid_rows);
  dim3 blockSize(BLOCK_SIZE_161616, BLOCK_SIZE_161616);
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