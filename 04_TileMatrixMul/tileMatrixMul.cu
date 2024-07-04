#include <Eigen/Dense>
#include <iostream>
#define TILE_WIDTH 16

__global__ void tileMatrixMulKernel(
    float *M, float *N, float *P, int Width) {

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int ph = 0; ph < ceil(float(Width)/TILE_WIDTH); ++ph) {
            // Collaborative loading of M and N tiles into shared memory
            if (Row < Width && ph * TILE_WIDTH + tx < Width)
                Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
            else 
                Mds[ty][tx] = 0.0;
            
            if (Col < Width && ph * TILE_WIDTH + ty < Width)
                Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
            else
                Nds[ty][tx] = 0.0;
            __syncthreads();

            // Perform tile matrix multiplication
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += Mds[ty][k] * Nds[k][tx];
                __syncthreads();
            }
    }

    if (Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;
}

int main() {
    // Size related
    int Width = 17;

    // Create two matrices M, N
    // Be careful that Eigen uses column-major order by default
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(Width, Width);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N(Width, Width);

    // Set the elements of M and N
    M.setOnes(); // set all elements to 1
    N.setOnes(); // set all elements to 1

    // Create pointer to the data
    float *M_ptr = M.data();
    float *N_ptr = N.data();

    // Create a matrix to store the result of M * N
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(Width, Width);
    float *P_ptr = P.data();
    
    // Allocate memory and move data to GPU
    int size = Width * Width * sizeof(float);
    float *M_d, *N_d, *P_d;
    cudaMalloc((void**)&M_d, size);
    cudaMalloc((void**)&N_d, size);
    cudaMalloc((void**)&P_d, size);
    cudaMemcpy(M_d, M_ptr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_ptr, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(ceil(Width/16.0), ceil(Width/16.0));
    tileMatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);

    // Copy the result back to the host
    cudaMemcpy(P_ptr, P_d, size, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << P << std::endl;

    // Free the memory
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return 0;
}