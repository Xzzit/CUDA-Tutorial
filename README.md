# CUDA-Tutorial

## Environment Setup
```
sudo apt-get update
sudo apt-get install libopencv-dev # OpenCV
sudo apt-get install libeigen3-dev # Eigen
```

## Hardware and Software Requirements
Tested with:

* Ubuntu 22.04
* GCC 11.3.0
* G++ 11.3.0
* CUDA capable GPU with compute capability 5.2 or later
* CUDA Toolkit 11.5 or later
* CUDA Driver 510 or later

## 3070 laptop specs
```
Performance Information
-----------------------
FP32 (float):   15.97 TFLOPS
FP64 (double):  249.6 GFLOPS
Bandwidth：     448.06 GB/s
-----------------------

Memory Information
-----------------------
Total global memory:    8 GB
Shared memory / block:  48 KB
Shared memory / SM:     100 KB
#registers / block:     65536
-----------------------

Limits
-----------------------
Max threads / block:     1024
Max threads / SM:        1536
```