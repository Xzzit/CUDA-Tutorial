# 04 Tiled Matrix Multiplication

## Introduction
This chapter is corresponding to the PMPP Chapter 05. The code is designed to implement tiled matrix multiplication based on the naive matrix multiplication approach by dividing the matrix into tiles, storing these tiles in shared memory, and then calculating the result.

## Compile and run the code
Require Eigen library.

```bash
# Compile
nvcc -I /usr/include/eigen3/ tileMatrixMul.cu -o tileMatrixMul

# Run
./tileMatrixMul
```