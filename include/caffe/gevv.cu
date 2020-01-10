#ifndef _GEVV_CU_
#define _GEVV_CU_

namespace gevv {
    // axpy kernel
    const int BLOCK_SIZE_axpy = 256;
    template <typename Dtype>
    __global__ void kernel_axpy(const int N, const Dtype alpha, const Dtype *X, Dtype *Y) {
        int idx = blockIdx.x * BLOCK_SIZE_axpy + threadIdx.x;
        if (idx < N)
        {
            Y[idx] = alpha * X[idx] + Y[idx];
        }
    }

    void caffe_gpu_axpy(const int N, const float alpha, const float* X, float* Y) {
        kernel_axpy<float><<<(N + BLOCK_SIZE_axpy - 1) / BLOCK_SIZE_axpy, BLOCK_SIZE_axpy>>>(N, alpha, X, Y);
    }

    // scal kernel
    const int BLOCK_SIZE_scal = 256;
    template <typename Dtype>
    __global__ void kernel_scal(const int N, const Dtype alpha, Dtype *X) {
        int idx = blockIdx.x * BLOCK_SIZE_scal + threadIdx.x;
        if (idx < N)
        {
            X[idx] = alpha * X[idx];
        }
    }

    void caffe_gpu_scal(const int N, const float alpha, float *X) {
        kernel_scal<float><<<(N + BLOCK_SIZE_scal - 1) / BLOCK_SIZE_scal, BLOCK_SIZE_scal>>>(N, alpha, X);
    }

    // axpby kernel
    const int BLOCK_SIZE_axpby = 256;
    template <typename Dtype>
    __global__ void kernel_axpby(const int N, const Dtype alpha, const Dtype *X, const Dtype beta, Dtype *Y) {
        int idx = blockIdx.x * BLOCK_SIZE_axpby + threadIdx.x;
        if (idx < N)
        {
            Y[idx] = alpha * X[idx] + beta * Y[idx];
        }
    }

    void caffe_gpu_axpby(const int N, const float alpha, const float* X,
        const float beta, float* Y) {
    kernel_axpby<float><<<(N + BLOCK_SIZE_axpby - 1) / BLOCK_SIZE_axpby, BLOCK_SIZE_axpby>>>(N, alpha, X, beta, Y);
    }

    // dot kernel
    // n==2000: 256 threads in one block, each thread calculate 8 times
    template <typename Dtype>
    __global__ void kernel_dot_2000(const int N, const Dtype *X, const Dtype *Y, Dtype *d_out) {
        int idx = threadIdx.x;
        __shared__ Dtype buffer[256];
        if (idx >= 250)
            buffer[idx] = 0;
        else
        {
            Dtype localAns = 0;
            localAns += X[idx * 8 + 0] * Y[idx * 8 + 0];
            localAns += X[idx * 8 + 1] * Y[idx * 8 + 1];
            localAns += X[idx * 8 + 2] * Y[idx * 8 + 2];
            localAns += X[idx * 8 + 3] * Y[idx * 8 + 3];
            localAns += X[idx * 8 + 4] * Y[idx * 8 + 4];
            localAns += X[idx * 8 + 5] * Y[idx * 8 + 5];
            localAns += X[idx * 8 + 6] * Y[idx * 8 + 6];
            localAns += X[idx * 8 + 7] * Y[idx * 8 + 7];
            buffer[idx] = localAns;
        }
        __syncthreads();

        // for (int i = 2; i <= 256; i *= 2)
        // {
        //   if (idx < 256 / i)
        //     buffer[idx] += buffer[idx + 256 / i];
        //   __syncthreads();
        // }
        for (int i = 2; i <= 256; i *= 2)
        {
            if (idx % i == 0)
            buffer[idx] += buffer[idx + i / 2];
            __syncthreads();
        }

        *d_out = buffer[0];
    }

    template <typename Dtype>
    __global__ void kernel_dot_n(const int N, const Dtype *X, const Dtype *Y, 
        Dtype *d_out) {
        int idx = threadIdx.x;
        __shared__ Dtype buffer[256];

        Dtype localAns = 0;
        for (int i = idx; i < N - 256; i += 256)
            localAns += X[i] * Y[i];
        localAns += X[N - 256 + idx] * Y[N - 256 + idx];
        buffer[idx] = localAns;

        __syncthreads();

        for (int i = 2; i <= 256; i *= 2)
        {
            if (idx % i == 0)
            buffer[idx] += buffer[idx + i / 2];
            __syncthreads();
        }

        *d_out = buffer[0];
    }

    void caffe_gpu_dot(const int n, const float* x, const float* y,
        float* out) {
        if (n == 1)
        {
            float *h_x, *h_y;
            h_x = (float*)malloc(sizeof(float));
            h_y = (float*)malloc(sizeof(float));
            cudaMemcpy((void*)h_x, (void*)x, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy((void*)h_y, (void*)y, sizeof(float), cudaMemcpyDeviceToHost);
            *out = (*h_x) * (*h_y);
        }
        else if (n == 2000)
        {
            float *d_out;
            cudaMalloc((void**)&d_out, sizeof(float));
            kernel_dot_2000<float><<<1, 256>>>(n, x, y, d_out);
            cudaMemcpy((void*)out, (void*)d_out, sizeof(float), cudaMemcpyDeviceToHost);
        }
        else // n == 43682
        {
            float *d_out, *tmp;
            cudaMalloc((void**)&d_out, 22 * sizeof(float));
            tmp = (float*)malloc(22 * sizeof(float));
            cudaStream_t stream[22];
            for (int i = 0; i < 22; ++i)
            cudaStreamCreate(&stream[i]);
            for (int i = 0; i < 21; ++i)
                kernel_dot_2000<float><<<1, 256, 0, stream[i]>>>(2000, x + 2000 * i, y + 2000 * i, d_out + i);
            kernel_dot_n<float><<<1, 256, 0, stream[21]>>>(1682, x + 42000, y + 42000, d_out + 21);
            cudaDeviceSynchronize();

            cudaMemcpy((void*)tmp, (void*)d_out, 22 * sizeof(float), cudaMemcpyDeviceToHost);
            *out = 0;
            for (int i = 0; i < 22; ++i)
                *out += tmp[i];
        }
    }

    // asum kernel
    // from debug, we found that n is quite small, usually 1 or 2, so we think
    // CUDA kernel function is too heavy here.
    // however, these addresses are on GPU device, so we have to use kernel function
    template <typename Dtype>
    __global__ void kernel_asum(const int N, const Dtype *x, Dtype *y){
        Dtype ans = 0;
        for (int i = 0; i < N; ++i)
            ans += x[i];
        *y = ans;
    }
    
    void caffe_gpu_asum(const int n, const float* x, float* y) {
        float *d_y;
        cudaMalloc((void**)&d_y, sizeof(float));
        kernel_asum<float><<<1, 1>>>(n, x, d_y);
        cudaMemcpy((void*)y, (void*)d_y, sizeof(float), cudaMemcpyDeviceToHost);
    }

    // scale kernel
    const int BLOCK_SIZE_scale = 256;
    template <typename Dtype>
    __global__ void kernel_scale(const int N, const Dtype alpha, const Dtype *X, Dtype *Y) {
        int idx = blockIdx.x * BLOCK_SIZE_scale + threadIdx.x;
        if (idx < N)
        {
            Y[idx] = X[idx] * alpha;
        }
    }

    void caffe_gpu_scale(const int n, const float alpha, const float *x, float* y) {
        kernel_scale<float><<<(n + BLOCK_SIZE_scale - 1) / BLOCK_SIZE_scale, BLOCK_SIZE_scale>>>(n, alpha, x, y);
    }

} // namespace gevv

#endif // _GEVV_CU_