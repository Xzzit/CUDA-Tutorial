#include <iostream>
#include <math.h>
#include <chrono>
using namespace std::chrono;

// Kernel function to add the elements of two arrays
__global__
void add_0(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride){
    y[i] = x[i] + y[i];
  }
}

/*
Another style for writting kernel function

Note: If grids contain many more blocks (numBlocks) than the limit
of total number of blocks that can be simultaneously executing in a CUDA
device, the runtime system maintains a list of blocks that need to execute
and assigns new blocks to SMs when previously assigned blocks complete execution.
*/
__global__
void add_1(int n, float *x, float *y)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i<n) {
        y[i] = x[i] + y[i];
    }
}

int main(void)
{
    int N = 1<<25;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Get starting timepoint
    auto start = high_resolution_clock::now();

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add_0<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Get ending timepoint
    auto stop = high_resolution_clock::now();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by function: "
    << duration.count() << " microseconds" << std::endl;
    
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