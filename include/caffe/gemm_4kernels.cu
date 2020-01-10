#ifndef _GEMM_CU_
#define _GEMM_CU_

namespace gemm {

#define BLOCK_SIZE 16
#define tile_a_x_tile_b (tile_a[ty][0]*tile_b[0][tx]+tile_a[ty][1]*tile_b[1][tx]+tile_a[ty][2]*tile_b[2][tx]+tile_a[ty][3]*tile_b[3][tx]+ \
  tile_a[ty][4]*tile_b[4][tx]+tile_a[ty][5]*tile_b[5][tx]+tile_a[ty][6]*tile_b[6][tx]+tile_a[ty][7]*tile_b[7][tx]+ \
  tile_a[ty][8]*tile_b[8][tx]+tile_a[ty][9]*tile_b[9][tx]+tile_a[ty][10]*tile_b[10][tx]+tile_a[ty][11]*tile_b[11][tx]+ \
  tile_a[ty][12]*tile_b[12][tx]+tile_a[ty][13]*tile_b[13][tx]+tile_a[ty][14]*tile_b[14][tx]+tile_a[ty][15]*tile_b[15][tx])


template <typename Dtype>
__global__ void kernel_gemm_nn_oo(int grid_rows_ii, int grid_cols_ii, const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii) * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = (col < N) ? b[(i + ty) * N + col] : 0;
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
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
__global__ void kernel_gemm_tn_oo(int grid_rows_ii, int grid_cols_ii, const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii) * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = (col < N) ? b[(i + ty) * N + col] : 0;
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
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
__global__ void kernel_gemm_nt_oo(int grid_rows_ii, int grid_cols_ii, const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii) * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = (col < N) ? b[col * K + i + ty] : 0;
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
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
__global__ void kernel_gemm_tt_oo(int grid_rows_ii, int grid_cols_ii, const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii) * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = (col < N) ? b[col * K + i + ty] : 0; //
    __syncthreads(); 
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0; //
  __syncthreads();
  ans += tile_a_x_tile_b;
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
__global__ void kernel_gemm_nn_oi(int grid_rows_ii, const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = b[(i + ty) * N + col];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();  

  ans *= alpha;
  if (row < M)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}

template <typename Dtype>
__global__ void kernel_gemm_tn_oi(int grid_rows_ii, const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = b[(i + ty) * N + col];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (row < M)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}

template <typename Dtype>
__global__ void kernel_gemm_nt_oi(int grid_rows_ii, const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[row * K + i + tx] : 0;
    tile_b[ty][tx] = b[col * K + i + ty];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[col * K + i + ty] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (row < M)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}

template <typename Dtype>
__global__ void kernel_gemm_tt_oi(int grid_rows_ii, const int M, const int N, const int K,
  const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
  Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = (blockIdx.y + grid_rows_ii) * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = b[col * K + i + ty]; //
    __syncthreads(); 
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (row < M && i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[col * K + i + ty] : 0; //
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (row < M)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}

































































template <typename Dtype>
__global__ void kernel_gemm_nn_io(int grid_cols_ii, const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii)* BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = a[row * K + i + tx];
    tile_b[ty][tx] = (col < N) ? b[(i + ty) * N + col] : 0;
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();  

  ans *= alpha;
  if (col < N)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}

template <typename Dtype>
__global__ void kernel_gemm_tn_io(int grid_cols_ii, const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii) * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = a[(i + tx) * M + row];
    tile_b[ty][tx] = (col < N) ? b[(i + ty) * N + col] : 0;
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (col < N)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}

template <typename Dtype>
__global__ void kernel_gemm_nt_io(int grid_cols_ii, const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii) * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = a[row * K + i + tx];
    tile_b[ty][tx] = (col < N) ? b[col * K + i + ty] : 0;
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (col < N)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}

template <typename Dtype>
__global__ void kernel_gemm_tt_io(int grid_cols_ii, const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = (blockIdx.x + grid_cols_ii) * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = a[(i + tx) * M + row];
    tile_b[ty][tx] = (col < N) ? b[col * K + i + ty] : 0; //
    __syncthreads(); 
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (col < N && i + ty < K) ? b[col * K + i + ty] : 0; //
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (col < N)
  {
    if (beta != 0)
      c[row * N + col] = c[row * N + col] * beta + ans;
    else
      c[row * N + col] = ans;
  }
}















































template <typename Dtype>
__global__ void kernel_gemm_nn_ii(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = a[row * K + i + tx];
    tile_b[ty][tx] = b[(i + ty) * N + col];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();  

  ans *= alpha;
  if (beta != 0)
    c[row * N + col] = c[row * N + col] * beta + ans;
  else
    c[row * N + col] = ans;
  
}

template <typename Dtype>
__global__ void kernel_gemm_tn_ii(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = (row < M) ? a[(i + tx) * M + row] : 0;
    tile_b[ty][tx] = (col < N) ? b[(i + ty) * N + col] : 0;
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[(i + ty) * N + col] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (beta != 0)
    c[row * N + col] = c[row * N + col] * beta + ans;
  else
    c[row * N + col] = ans;
  
}

template <typename Dtype>
__global__ void kernel_gemm_nt_ii(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = a[row * K + i + tx];
    tile_b[ty][tx] = b[col * K + i + ty];
    __syncthreads();
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[row * K + i + tx] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[col * K + i + ty] : 0;
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (beta != 0)
    c[row * N + col] = c[row * N + col] * beta + ans;
  else
    c[row * N + col] = ans;
  
}

template <typename Dtype>
__global__ void kernel_gemm_tt_ii(const int M, const int N, const int K,
    const Dtype alpha, const Dtype *a, const Dtype *b, const Dtype beta,
    Dtype *c) {
  __shared__ Dtype tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ Dtype tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * BLOCK_SIZE + ty;
  int col = blockIdx.x * BLOCK_SIZE + tx;
  Dtype ans = 0;

  int i;
  for (i = 0; i < K - BLOCK_SIZE; i += BLOCK_SIZE)
  {
    tile_a[ty][tx] = a[(i + tx) * M + row];
    tile_b[ty][tx] = b[col * K + i + ty];
    __syncthreads(); 
    ans += tile_a_x_tile_b;
    __syncthreads();
  }

  tile_a[ty][tx] = (i + tx < K) ? a[(i + tx) * M + row] : 0;
  tile_b[ty][tx] = (i + ty < K) ? b[col * K + i + ty] : 0; //
  __syncthreads();
  ans += tile_a_x_tile_b;
  __syncthreads();

  ans *= alpha;
  if (beta != 0)
    c[row * N + col] = c[row * N + col] * beta + ans;
  else
    c[row * N + col] = ans;
  
}












































void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {

      int grid_rows_ii = M / BLOCK_SIZE;
      int grid_cols_ii = N / BLOCK_SIZE;

      // printf("grid_rows_ii = %d\n",grid_rows_ii);
      // printf("grid_cols_ii = %d\n",grid_cols_ii);
      // printf("%d x %d\n", M, N);

      int grid_rows_oi = (int)(grid_rows_ii * BLOCK_SIZE != M);
      int grid_cols_oi = grid_cols_ii * grid_rows_oi;

      int grid_cols_io = (int)(grid_cols_ii * BLOCK_SIZE != N);
      int grid_rows_io = grid_rows_ii * grid_cols_io;

      int grid_rows_oo = grid_rows_oi;
      int grid_cols_oo = grid_cols_io;

      dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

      dim3 gridSize_ii(grid_cols_ii, grid_rows_ii);          
      dim3 gridSize_io(grid_cols_io, grid_rows_io);
      dim3 gridSize_oi(grid_cols_oi, grid_rows_oi);
      dim3 gridSize_oo(grid_cols_oo, grid_rows_oo);

      // cudaStream_t stream[4];
      // for(int i = 0; i < 4; i++)
      // {
      //   cudaStreamCreate(&stream[i]);
      // }

      if (TransA == CblasNoTrans && TransB == CblasNoTrans) {

          if (grid_cols_ii != 0 && grid_rows_ii != 0)
          kernel_gemm_nn_ii<float><<<gridSize_ii, blockSize>>>(M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nn_ii<float><<<gridSize_ii, blockSize, 0, stream[0]>>>(M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io != 0 && grid_rows_ii != 0)
          kernel_gemm_nn_io<float><<<gridSize_io, blockSize>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nn_io<float><<<gridSize_io, blockSize, 0, stream[1]>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_rows_oi != 0 && grid_cols_ii != 0)
          kernel_gemm_nn_oi<float><<<gridSize_oi, blockSize>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nn_oi<float><<<gridSize_oi, blockSize, 0, stream[2]>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io && grid_cols_oi)
          kernel_gemm_nn_oo<float><<<gridSize_oo, blockSize>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nn_oo<float><<<gridSize_oo, blockSize, 0, stream[3]>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);

      } else if (TransA != CblasNoTrans && TransB == CblasNoTrans) {

          if (grid_cols_ii != 0 && grid_rows_ii != 0)
          kernel_gemm_tn_ii<float><<<gridSize_ii, blockSize>>>(M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tn_ii<float><<<gridSize_ii, blockSize, 0, stream[0]>>>(M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io != 0 && grid_rows_ii != 0)
          kernel_gemm_tn_io<float><<<gridSize_io, blockSize>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tn_io<float><<<gridSize_io, blockSize, 0, stream[1]>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_rows_oi != 0 && grid_cols_ii != 0)
          kernel_gemm_tn_oi<float><<<gridSize_oi, blockSize>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tn_oi<float><<<gridSize_oi, blockSize, 0, stream[2]>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io && grid_cols_oi)
          kernel_gemm_tn_oo<float><<<gridSize_oo, blockSize>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tn_oo<float><<<gridSize_oo, blockSize, 0, stream[3]>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);

      } else if (TransA == CblasNoTrans && TransB != CblasNoTrans) {

          if (grid_cols_ii != 0 && grid_rows_ii != 0)
          kernel_gemm_nt_ii<float><<<gridSize_ii, blockSize>>>(M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nt_ii<float><<<gridSize_ii, blockSize, 0, stream[0]>>>(M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io != 0 && grid_rows_ii != 0)
          kernel_gemm_nt_io<float><<<gridSize_io, blockSize>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nt_io<float><<<gridSize_io, blockSize, 0, stream[1]>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_rows_oi != 0 && grid_cols_ii != 0)
          kernel_gemm_nt_oi<float><<<gridSize_oi, blockSize>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nt_oi<float><<<gridSize_oi, blockSize, 0, stream[2]>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io && grid_cols_oi)
          kernel_gemm_nt_oo<float><<<gridSize_oo, blockSize>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_nt_oo<float><<<gridSize_oo, blockSize, 0, stream[3]>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          
      } else {

          if (grid_cols_ii != 0 && grid_rows_ii != 0)
          kernel_gemm_tt_ii<float><<<gridSize_ii, blockSize>>>(M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tt_ii<float><<<gridSize_ii, blockSize, 0, stream[0]>>>(M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io != 0 && grid_rows_ii != 0)
          kernel_gemm_tt_io<float><<<gridSize_io, blockSize>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tt_io<float><<<gridSize_io, blockSize, 0, stream[1]>>>(grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_rows_oi != 0 && grid_cols_ii != 0)
          kernel_gemm_tt_oi<float><<<gridSize_oi, blockSize>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tt_oi<float><<<gridSize_oi, blockSize, 0, stream[2]>>>(grid_rows_ii, M, N, K, alpha, A, B, beta, C);
          
          if (grid_cols_io && grid_cols_oi)
          kernel_gemm_tt_oo<float><<<gridSize_oo, blockSize>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          // kernel_gemm_tt_oo<float><<<gridSize_oo, blockSize, 0, stream[3]>>>(grid_rows_ii, grid_cols_ii, M, N, K, alpha, A, B, beta, C);
          
      }

      cudaError_t er1 = cudaPeekAtLastError();
      CUDA_CHECK(er1);

      // for(int i = 0; i < 4; i++)
      // {
      //   cudaStreamSynchronize(stream[i]);
      // }
}

}
#endif // _GEMM_CU_