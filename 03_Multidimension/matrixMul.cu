#include <Eigen/Dense>
#include <iostream>

__global__ void matrixMulKernel(
    float *M, float *N, float *P, 
    int I, int J, int K) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < I && j < K) {
        float sum = 0;
        for (int k = 0; k < J; k++) {
            sum += M[i * J + k] * N[k * K + j];
        }
        P[i * K + j] = sum;
    }
}

int main() {
    // Size related
    int I = 2;
    int J = 3;
    int K = 2;

    // Create two matrices M(IxJ), N(JxK)
    // Be careful that Eigen uses column-major order by default
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(I, J);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N(J, K);

    // Set the elements of M and N
    M.setOnes(); // set all elements to 1
    N.setConstant(2); // set all elements to 2
    // M << 1, 2, 3, 4, 5, 6; # 2x3
    // N << 1, 2, 3, 4, 5, 6; # 3x2

    // Create pointer to the data
    float *M_ptr = M.data();
    float *N_ptr = N.data();

    // Create a matrix to store the result of M * N
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(I, K);
    float *P_ptr = P.data();
    
    // Allocate memory and move data to GPU
    int size_M = I * J * sizeof(float);
    int size_N = J * K * sizeof(float);
    int size_P = I * K * sizeof(float);
    float *M_d, *N_d, *P_d;
    cudaMalloc((void**)&M_d, size_M);
    cudaMalloc((void**)&N_d, size_N);
    cudaMalloc((void**)&P_d, size_P);
    cudaMemcpy(M_d, M_ptr, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_ptr, size_N, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(I/16.0), ceil(K/16.0));
    matrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, I, J, K);

    // Copy the result back to the host
    cudaMemcpy(P_ptr, P_d, size_P, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << P << std::endl;

    // Free the memory
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}