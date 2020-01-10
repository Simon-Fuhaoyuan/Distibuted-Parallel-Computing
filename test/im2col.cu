
const int BLOCK_SIZE = 8;


template <typename Dtype>
__global__ void im2col_gpu_kernel(
    const int yBlocks_per_channel,
    const Dtype* data_im,
    const int height, const int width, 
    
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,

    const int height_col, const int width_col,
    Dtype* data_col
) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int c_im = by / yBlocks_per_channel;
    int h_im = (by % yBlocks_per_channel) * BLOCK_SIZE + ty;
    int w_im = bx * BLOCK_SIZE + tx;

    __shared__ Dtype shmm[BLOCK_SIZE][BLOCK_SIZE];
    if (h_im < height && w_im < width) {
        shmm[ty][tx] = data_im[
            c_im * width * height + h_im * width + w_im];
    }
    __syncthreads();
      
    // padding
    h_im -= pad_h;
    w_im -= pad_w;
    ty -= pad_h;
    tx -= pad_w;

    // copy adjacent elements of (ty, tx) from `shmm` to `data_col` 
    // if (h_im, w_im) is a pixel to be convoluted.
    // dilation is assumed to be 1.
    if (h_im < height + pad_h && w_im < width + pad_w && 
        (h_im + pad_h) % stride_h == 0 && 
        (w_im + pad_w) % stride_w == 0
    ) {
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                data_col[
                    c_im * height_col * width_col + 
                    (h_im + pad_h) / stride_h * width_col + 
                    (w_im + pad_w) / stride_w
                ] = 
                    (
                        ty + i >= 0 && tx + j >= 0 && 
                        ty + i < BLOCK_SIZE && tx + j < BLOCK_SIZE && 
                        h_im + i < height + pad_h && w_im + j <= width + pad_w
                    ) ? shmm[ty + i][tx + j] : 0;
            }
        }
    }
}

//
template <typename Dtype>
void im2col_gpu(
    const Dtype* data_im, 
    const int channels, const int height, const int width, 

    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,

    Dtype* data_col
) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h -
        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w -
        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    int yBlocks_per_channel = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_rows = yBlocks_per_channel * channels;
    int grid_cols = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 gridSize(grid_cols, grid_rows);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    
    im2col_gpu_kernel<Dtype><<<gridSize, blockSize>>>(
        yBlocks_per_channel,
        data_im, height, width, kernel_h, kernel_w, pad_h,
        pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
        width_col, data_col);
    CUDA_POST_KERNEL_CHECK;
}