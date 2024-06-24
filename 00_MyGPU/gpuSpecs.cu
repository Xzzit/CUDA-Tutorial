#include <cuda_runtime_api.h>
#include <iostream>
#define WINDOW_SIZE 73

/*
Before you use your GPU to do work, you should know the 
most essential things about its capabilities.
*/
int main()
{
	// Count CUDA-capable devices on the system
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	if (numDevices == 0)
	{
		std::cout << "You have no CUDA devices available!" << std::endl;
		return -1;
	}

	// Iterate through devices
	cudaDeviceProp props;
	for (unsigned int i=0; i<numDevices; i++) {
		cudaGetDeviceProperties(&props, i);

		/* 
		We only print the most fundamental properties here. cudaDeviceProp 
		contains a long range of indicators to check for different things
		that your GPU may or may not support, as well as factors for 
		performance. However, the most essential property to know about is
		the compute capability of the device. 
		*/
		std::cout << std::string(WINDOW_SIZE, '-') << std::endl;

		std::cout << "Model: " << props.name << std::endl;
		std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
		
		std::cout << "Multiprocessors: " << props.multiProcessorCount << std::endl;
		std::cout << "Memory: " << props.totalGlobalMem / float(1 << 30) << " GiB" << std::endl;
		std::cout << "Clock rate: " << props.clockRate / float(1'000'000) << " GHz" << std::endl;
		std::cout << "L2 Cache Size: " << props.l2CacheSize / float(1 << 20)<< " MB" << std::endl;

		printf("Numbers of registers available per block:      %d\n",
			props.regsPerBlock);
		printf("Shared memory available per block:             %zu\n",
			props.sharedMemPerBlock);
		printf("Shared memory available per multiprocessor:    %zu\n",
			props.sharedMemPerMultiprocessor);
		printf("Maximum number of threads per multiprocessor:  %d\n",
			props.maxThreadsPerMultiProcessor);
		printf("Maximum number of threads per block:           %d\n",
			props.maxThreadsPerBlock);
		printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
		printf("Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
		printf("Warp size:  %d\n", props.warpSize);

		std::cout << std::string(WINDOW_SIZE, '-') << std::endl;
	}
	
	return 0;
}