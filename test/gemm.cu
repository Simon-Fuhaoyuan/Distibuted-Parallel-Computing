// gemm kernel
const int BLOCK_SIZE = 16;
	
template <typename Dtype>
__global__ void kernel_gemm_nn(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  Dtype ans = 0;

  for (int i = 0; i < K; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j)
      ans += tile_a[ty][j] * tile_b[j][tx];
    __syncthreads();
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
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  Dtype ans = 0;

  for (int i = 0; i < K; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j)
      ans += tile_a[ty][j] * tile_b[j][tx];
    __syncthreads();
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
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  Dtype ans = 0;

  for (int i = 0; i < K; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0;
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j)
      ans += tile_a[ty][j] * tile_b[j][tx];
    __syncthreads();
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
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  Dtype ans = 0;

  for (int i = 0; i < K; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0;
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j)
      ans += tile_a[ty][j] * tile_b[j][tx];
    __syncthreads();
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

// gemm interface
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  // CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
  //     N, M, K, &alpha, B, ldb, A, lda, &beta, C, N)); //
  
  int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 gridSize(grid_cols, grid_rows);
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  if (TransA == CblasNoTrans && TransB == CblasNoTrans)
    kernel_gemm_nn<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
  else if (TransA != CblasNoTrans && TransB == CblasNoTrans)
    kernel_gemm_tn<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
  else if (TransA == CblasNoTrans && TransB != CblasNoTrans)
    kernel_gemm_nt<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
  else
    kernel_gemm_tt<float><<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
}
