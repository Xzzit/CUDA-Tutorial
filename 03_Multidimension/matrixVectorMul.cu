#include <Eigen/Dense>
#include <iostream>

__global__ void matrixMulKernel(
    float *M, float *v, float *p, 
    int Width) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < Width) {
        float sum = 0;
        for (int k = 0; k < Width; k++) {
            sum += M[i * Width + k] * v[k];
        }
        p[i] = sum;
    }
}

int main() {
    // Size related
    int Width = 6;

    // Create matrix and vector M, v
    // Be careful that Eigen uses column-major order by default
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(Width, Width);
    Eigen::VectorXf v(Width);

    // Set the elements of M and N
    M.setOnes(); // set all elements to 1
    v.setConstant(2); // set all elements to 2

    // Create pointer to the data
    float *M_ptr = M.data();
    float *v_ptr = v.data();

    // Create a vector to store the result of Mv
    Eigen::VectorXf p(Width);
    float *p_ptr = p.data();
    
    // Allocate memory and move data to GPU
    int size_M = Width * Width * sizeof(float);
    int size_v = Width * sizeof(float);
    int size_p = Width * sizeof(float);
    float *M_d, *v_d, *p_d;
    cudaMalloc((void**)&M_d, size_M);
    cudaMalloc((void**)&v_d, size_v);
    cudaMalloc((void**)&p_d, size_p);
    cudaMemcpy(M_d, M_ptr, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_ptr, size_v, cudaMemcpyHostToDevice);

    // Launch the kernel
    matrixMulKernel<<<ceil(Width/16.0), 16>>>(M_d, v_d, p_d, Width);

    // Copy the result back to the host
    cudaMemcpy(p_ptr, p_d, size_p, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << p << std::endl;

    // Free the memory
    cudaFree(M_d);
    cudaFree(v_d);
    cudaFree(p_d);

    return 0;
}