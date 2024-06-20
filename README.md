# CUDA-Tutorial

## Environment Setup
```
sudo apt-get update
sudo apt-get install libopencv-dev # OpenCV
sudo apt-get install libeigen3-dev # Eigen
```

## Compile and run the code
```
# Chapter 00~02
nvcc main.cu -o main
./main

# Chapter 03
nvcc your_code.cu $(pkg-config --cflags --libs opencv4) -o your_code
nvcc -I /usr/include/eigen3/ matrixMul.cu -o matrixMul
./your_code
```


Tested with:

* Ubuntu 22.04
* GCC 11.3.0
* G++ 11.3.0
* CUDA capable GPU with compute capability 5.2 or later
* CUDA Toolkit 11.5 or later
* CUDA Driver 510 or later
