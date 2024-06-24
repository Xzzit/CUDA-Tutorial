#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate memory and move data to GPU
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Get starting timepoint
    auto start = std::chrono::high_resolution_clock::now();

    // Part 2: Kernel launch
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // Get ending timepoint
    auto stop = std::chrono::high_resolution_clock::now();

    // Part 3: Copy data back to CPU and free memory
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // Get duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: "
    << duration.count() << " microseconds" << std::endl;
}

int main() {

    int n = 1<<25;

    std::vector<float> A(n, 1.0f);
    std::vector<float> B(n, 2.0f);
    std::vector<float> C(n);

    vecAdd(A.data(), B.data(), C.data(), n);

    std::vector<float> D(n, 3.0f);
    if (C == D) {
        printf("Success!\n");
    } else {
        printf("Error!\n");
    }

    return 0;
}


/*
Exercises:

1) Experiment with printf() inside the kernel. 
Try printing out the values of threadIdx.xand blockIdx.x for some or all of the threads. 
Do they print in sequential order? Why or why not?

2) Print the value of threadIdx.y or threadIdx.z (or blockIdx.y) in the kernel. 
(Likewise for blockDim and gridDim). Why do these exist? 
How do you get them to take on values other than 0 (1 for the dims)?
*/