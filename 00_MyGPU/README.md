# 00 GPU Specs

## Introduction
This chapter is corresponding to the PMPP Chapter 04. In this chapter, we will learn how to get the GPU specs using CUDA. The GPU specs include the number of SMs, the number of CUDA cores, the clock rate, the memory size, and the memory bandwidth and so on.

## Compile and run the code
```bash
nvcc gpuSpecs.cu -o gpuSpecs
./gpuSpecs
```