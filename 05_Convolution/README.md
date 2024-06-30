# 05 Convolution

## Introduction
This chapter is corresponding to the PMPP Chapter 07. In this chapter, we will learn how to implement convolution using CUDA. The task involves: 2D convolution and 2D convolution with constant memory.

In `conv.cu`, we demonstrate a basic 2D convolution. The code reveals we're using just 1% of the GPU's power, limited by slow bandwidth.

## Compile and run the code
Require OpenCV library.

```bash
# Compile
nvcc conv.cu $(pkg-config --cflags --libs opencv4) -I /usr/include/eigen3/ -o conv
nvcc convConstMem.cu $(pkg-config --cflags --libs opencv4) -I /usr/include/eigen3/ -o convConstMem
nvcc convTile.cu $(pkg-config --cflags --libs opencv4) -I /usr/include/eigen3/ -o convTile

# Run
./conv
./convConstMem
./convTile
```