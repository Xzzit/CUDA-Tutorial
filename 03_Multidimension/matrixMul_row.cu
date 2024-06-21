#include <Eigen/Dense>
#include <iostream>

__global__ void matrixMulKernel(
    float *M, float *N, float *P, 
    int Width) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Width) {
        for (int j = 0; j < Width; j++) {
            float sum = 0;
            for (int k = 0; k < Width; k++) {
                sum += M[i * Width + k] * N[k * Width + j];
            }
            P[i * Width + j] = sum;
        }
    }
}

int main() {
    // Size related
    int Width = 1000;

    // Create two matrices M(IxJ), N(JxK)
    // Be careful that Eigen uses column-major order by default
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(Width, Width);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N(Width, Width);

    // Set the elements of M and N
    M.setOnes(); // set all elements to 1
    N.setConstant(2); // set all elements to 2
    // M << 1, 2, 3, 4, 5, 6; # 2x3
    // N << 1, 2, 3, 4, 5, 6; # 3x2

    // Create pointer to the data
    float *M_ptr = M.data();
    float *N_ptr = N.data();

    // Create a matrix to store the result of M * N
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(Width, Width);
    float *P_ptr = P.data();
    
    // Allocate memory and move data to GPU
    int size_M = Width * Width * sizeof(float);
    int size_N = Width * Width * sizeof(float);
    int size_P = Width * Width * sizeof(float);
    float *M_d, *N_d, *P_d;
    cudaMalloc((void**)&M_d, size_M);
    cudaMalloc((void**)&N_d, size_N);
    cudaMalloc((void**)&P_d, size_P);
    cudaMemcpy(M_d, M_ptr, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_ptr, size_N, cudaMemcpyHostToDevice);

    // Launch the kernel
    matrixMulKernel<<<ceil(Width/16.0), 16>>>(M_d, N_d, P_d, Width);

    // Copy the result back to the host
    cudaMemcpy(P_ptr, P_d, size_P, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << P_ptr[0] << std::endl;

    // Free the memory
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}


/*
Exercises:

1) Analyze the pros and cons of each of the two kernel designs.

2) Write a kernel that has each thread produce one output matrix column.
*/