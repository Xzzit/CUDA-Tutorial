#include <vector>
#include <stdio.h>
#include <iostream>

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

    // Part 2: Kernel launch
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // Part 3: Copy data back to CPU and free memory
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {

    int n;
    std::cout << "Enter the size of the array: ";
    std::cin >> n;

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