# 03 Multidimension

## Introduction
This chapter is corresponding to the PMPP Chapter 03. In this chapter, we will learn how to call a multi-dimensional CUDA kernel. The task involves: image blurring, RGB image to grayscale, matrix-matrix multiplication, and matrix-vector multiplication.

## Compile and run the code
Require OpenCV and Eigen library.

```bash
# Compile
nvcc grayscale.cu $(pkg-config --cflags --libs opencv4) -o grayscale
nvcc blur.cu $(pkg-config --cflags --libs opencv4) -o blur
nvcc -I /usr/include/eigen3/ matrixMul.cu -o matrixMul
nvcc -I /usr/include/eigen3/ matrixMul_row.cu -o matrixMul_row
nvcc -I /usr/include/eigen3/ matrixVectorMul.cu -o matrixVectorMul

# Run
./your_code
```