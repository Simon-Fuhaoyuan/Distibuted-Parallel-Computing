# CS433 Distributed and Parallel Computing Course
This repo includes some codes about 
> OpenMP \
> CUDA \
> Our final project -- **Optimizing Caffe**

The first two homework are quite naive, but they gave us a basic feeling of parallel programming, especially using CUDA to write kernel functions. The ability of writing high-performanced CUDA kernel functions, as our teacher said, decides whether you can get a satisfying offer!

And our final project is quite difficult. As we all know, Caffe is a state-of-art deep-learning structure. Though tensorflow and pytorch are more frequently used these years, you have to adimitted Caffe is an awesome structure producing its own system of deep learning, getting rather high performance with the help of CUDA kernel functions, cudnn and cublas library, all designed by NVIDIA, and letting deep-learning learners easy to use. Since we are still working on this project, our modified code will be pushed later(on about Jan 5th, 2020, our project deadline).

## Update: 2020.1.11 
Now our code for final project **Optimizing Caffe** has been pushed. There's one thing to mention: we actually optimize caffe only for the training process of [ResNet101](https://github.com/KaimingHe/deep-residual-networks) and [InceptionBN](https://github.com/pertusa/InceptionBN-21K-for-Caffe). In fact, caffe is a rather high-performance deep-learning structure, since it uses cublas function. The cublas funcion is not open source, and it is actually optimized based on NVIDIA GPU hardware, so our kernel function may not be able to get higher performance.

## Some instructions about our code

### Start
We only push our modified codes here. To get the full caffe source code, please goto home page of [Caffe](https://github.com/BVLC/caffe).