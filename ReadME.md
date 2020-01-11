# CS433 Distributed and Parallel Computing Course
This repo includes some codes about 
- OpenMP
- CUDA
- Our final project -- **Optimizing Caffe**

The first two homework are quite naive, but they gave us a basic feeling of parallel programming, especially using CUDA to write kernel functions. The ability of writing high-performanced CUDA kernel functions, as our teacher said, decides whether you can get a satisfying offer!

And our final project is quite difficult. As we all know, Caffe is a state-of-art deep-learning structure. Though tensorflow and pytorch are more frequently used these years, you have to adimitted Caffe is an awesome structure producing its own system of deep learning, getting rather high performance with the help of CUDA kernel functions, cudnn and cublas library, all designed by NVIDIA, and letting deep-learning learners easy to use. Since we are still working on this project, our modified code will be pushed later(on about Jan 5th, 2020, our project deadline).

## Update: 2020.1.11 
Now our code for final project **Optimizing Caffe** has been pushed. There's one thing to mention: we actually optimize caffe only for the training process of [ResNet101](https://github.com/KaimingHe/deep-residual-networks) and [InceptionBN](https://github.com/pertusa/InceptionBN-21K-for-Caffe). In fact, caffe is a rather high-performance deep-learning structure, since it uses cublas function. The cublas funcion is not open source, and it is actually optimized based on NVIDIA GPU hardware, so our kernel function may not be able to get higher performance.

## Quick start
### Installation
Clone our project to your workspace:
```
    git clone https://github.com/Simon-Fuhaoyuan/Distibuted-Parallel-Computing.git
```
Note that we only push our modified codes here. To get the full Caffe source code, please goto [home page of Caffe](https://github.com/BVLC/caffe). After getting full source code, please copy our *dataset* directory and *test* directory to the path of Caffe, and copy files under *include/caffe* to the same directory in source code, and update the *src/caffe/util/math_functions.cu* to our *src/caffe/util/math_functions.cu*.

### Data Preparation
Here we provide the dataset we used in our project to test the performance. However, you can set your own dataset using scripts of dataset in Caffe, and don't forget to converse the dataset into *lmdb* format and calculating the mean and std. We only use our dataset as an example here.

### Compile Caffe
After configuration of codes and dataset, you can compile the caffe source code using Makefile in Caffe. Please read [compiling instructions](http://caffe.berkeleyvision.org/installation.html#compilation) in detail and make sure your compilation is successful.

### Training
- **Note**: Our training process uses our small dataset including only 4 pictures about cats and dogs, respectively.

To train ResNet101, please use command:
```
    sh test/train_resnet_101.sh
```
To train InceptionBN, please use command:
```
    sh test/inception21k_train.sh
```

## Introduction about our optimization strategy
In summary, we use the following optimization strategy
- Data Tiling
- Shared Memory
- Loop Unrolling
- Stream Parallel
- Constant Memory
- Kernel Fusion
- Cache Friendly
- Optimizing case by case
For different tasks, different strategies are used, and different strategies are fit. Of all the strategies listed above, the last strategy is the most important -- optimizing case by case. When we used **gdb** to watch the matrix multiplication parameters, we found there're many multiplication sizes like 'M * 1' mutiplies '1 * N', so we also optimized such scenarios. According to different M and N, we may use different tiling width, block size and grid size.

## Final results
- **Note**: our results are tested under the situation of
```
    batch_size = 2
    max_iter = 100
```
 |ResNet101|InceptionBN
 --|:--|:--:
 Original Caffe|10.5iters/s|15.0iters/s
 Our Modified Caffe|5.0iters/s|8.4iters/s
 