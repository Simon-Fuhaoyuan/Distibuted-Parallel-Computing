#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/gemm.hpp"
#include "caffe/gemv.cu"
#include "caffe/gevv.cu"

// #include <bits/stdc++.h>
// using namespace std;

namespace caffe {

#define use_my_gemm

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {

      #ifdef use_my_gemm 
      // gemm kernel resnet batch = 2
          // blocksize:
              // 4: 1.3
              // 7: 2.7
              // 8: 4
              // 14: 4.15
              // 16: 5.08, 
              //    pragma unroll(4): 5.18
              //    manually unroll: 5.25
              //    unroll last block: 5.31
              //    with K1gemm: 5.31
              //    K1gemm without shmm: 5.31
              // 18: 4.5
              // 20: 4.5
              // 32: 4.85
              // cublas: 10.5
            
          if (
            (M == 2048 && N == 49 && K == 1) ||
            (M == 1024 && N == 49 && K == 2048) || 
            (M == 21841 && N == 1024 && K == 2)
          ) {
            gemm8::caffe_gpu_gemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
          } else if (
            K == 1
          ) {
            K1gemm::caffe_gpu_gemm(M, N, alpha, A, B, beta, C);
          } else if (
            (M == 256 && N == 2306 && K == 196) || 
            (M == 128 && N == 784 & K == 512) || 
            (M == 256 && N == 3136 && K == 64) ||
            (M == 64 && N == 3136 && K == 256) || 
            (M == 128 && N == 784 && K == 1152) || 
            (M == 128 && N == 1152 && K == 784) || 
            (M == 64 && N == 3136 && K == 576) || 
            (M == 2048 && N == 512 && K == 49) || 
            (M == 512 && N == 4608 && K == 49) || 
            (M == 4608 && N == 49 && K == 512) || 
            (M == 576 && N == 3136 && K == 64) ||
            (M == 64 && N == 3136 && K == 64) || 
            (M == 512 && N == 2048 && K == 49) || 
            (M == 64 && N == 12544 && K == 147) || 
            (M == 128 && N == 784 && K == 256) || 
            (M == 256 && N == 196 && K == 512) || 
            (M == 2048 && N == 49 && K == 1024) || 
            (M == 512 && N == 1024 && K == 49) || 
            (M == 2048 && N == 1024 && K == 49) || 
            (M == 256 && N == 512 && K == 196) || 
            (M == 512 && N == 256 && K == 784) || 
            (M == 128 && N == 196 && K == 576) || 
            (M == 128 && N == 576 && K == 196) || 
            (M == 144 && N == 864 && K == 784) || 
            (M == 384 && N == 784 && K == 96) || 
            (M == 96 && N == 784 && K == 288) || 
            (M == 96 && N == 576 && K == 196) || 
            (M == 224 && N == 2016 && K == 49) || 
            (M == 192 && N == 1024 && K == 49) || 
            (M == 320 && N == 1728 && K == 49) || 
            (M == 1728 && N == 49 && K == 320) || 
            (M == 128 && N == 608 && K == 196) || 
            (M == 96 && N == 864 && K == 196) || 
            (M == 864 && N == 196 && K == 96) || 
            (M == 96 && N == 12544 && K == 147) || 
            (M == 128 && N == 3136 && K == 96) || 
            (M == 288 && N == 3136 && K == 1152) || 
            (M == 96 && N == 784 && K == 864) || 
            (M == 96 && N == 784 && K == 480) || 
            (M == 128 && N == 196 && K == 512) || 
            (M == 256 && N == 196 && K == 1728) || 
            (M == 352 && N == 49 && K == 960) || 
            (M == 352 && N == 49 && K == 1024) || 
            (M == 128 && N == 1024 && K == 49) || 
            (M == 1024 && N == 49 && K == 128) || 
            (M == 1728 && N == 49 && K == 224) || 
            (M == 352 && N == 1024 && K == 49) || 
            (M == 1024 && N == 49 && K == 352) || 
            (M == 128 && N == 960 && K == 49) || 
            (M == 192 && N == 960 && K == 49) || 
            (M == 352 && N == 960 && K == 49) || 
            (M == 256 && N == 2304 && K == 49) || 
            (M == 512 && N == 196 && K == 192) || 
            (M == 192 && N == 1152 && K == 49) || 
            (M == 512 && N == 196 && K == 128) || 
            (M == 96 && N == 1440 && K == 196) || 
            (M == 608 && N == 196 && K == 160) || 
            (M == 608 && N == 196 && K == 96)
          ) {
              gemm32::caffe_gpu_gemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
          } else {
              gemm16::caffe_gpu_gemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
          } 
      #else 
          // Note that cublas follows fortran order.
          int lda = (TransA == Cbgemm16::caffe_gpu_gemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);lasNoTrans) ? K : M;
          int ldb = (TransB == CblasNoTrans) ? N : K;
          cublasOperation_t cuTransA =
              (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
          cublasOperation_t cuTransB =
              (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
          CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
              N, M, K, &alpha, B, ldb, A, lda, &beta, C, N)); 
      #endif 
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  gemv::caffe_gpu_gemv(TransA, M, N, alpha, A, x, beta, y);
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  gevv::caffe_gpu_axpy(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  gevv::caffe_gpu_scal(N, alpha, X);
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {
  cudaStream_t initial_stream;
  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  gevv::caffe_gpu_axpby(N, alpha, X, beta, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  gevv::caffe_gpu_dot(n, x, y, out);
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  gevv::caffe_gpu_asum(n, x, y);
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  gevv::caffe_gpu_scale(n, alpha, x, y);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

// cublas functions end

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

//
template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void sqrt_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = sqrt(a[index]);
  }
}

template <>
void caffe_gpu_sqrt<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_sqrt<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sqrt_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
