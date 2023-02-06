#include <iostream>

__global__ 
void cuda_hello()
{
    printf("Hello World from GPU!\n");
}


int main()
{
    printf("Hello World from CPU!\n");

    // call a kernel
    cuda_hello<<<1,1>>>(); 

    /*
	 Synchronize with GPU to wait for printf to finish.
	 Results of printf are buffered and copied back to
	 the CPU for I/O after the kernel has finished.
	*/
    cudaDeviceSynchronize();

    return 0;
}

/*
Exercises:
1) Change the message that is printed by the kernel
2) Write a different kernel (different name, different message)
3) Call the different kernels multiple times
*/