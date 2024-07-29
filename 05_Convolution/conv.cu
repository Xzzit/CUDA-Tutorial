#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#define CHANNEL 3

__global__ void convKernel(
    unsigned char *blur, unsigned char *bgr, float *kernel,
    int r, int height, int width) {

    // Indexes for output image
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    if (Col < width && Row < height) {
        int num_pix = 0;
        int val_b = 0;
        int val_g = 0;
        int val_r = 0;

        for (int kRow = 0; kRow < 2 * r + 1; kRow++) {
            for (int kCol = 0; kCol < 2 * r + 1; kCol++) {
                int idx = ((Row - r + kRow) * width + (Col - r + kCol)) * CHANNEL;
                if (idx >= 0 && idx < height * width * CHANNEL) {
                    /*
                    bgr: 1 byte, kernel: 4 bytes

                    Operation/Byte:     6 / 15 = 0.4 FLOP/B
                    Max Bandwidth:      448.0 GB/s
                    GFLOP/s:            448.0 * 0.4 = 179.2 GFLOP/s
                    Max Performance:    15.97 TFLOP/s
                    Utilization:        1.12%
                    */
                    val_b += bgr[idx] * kernel[kRow * (2 * r + 1) + kCol];
                    val_g += bgr[idx + 1] * kernel[kRow * (2 * r + 1) + kCol];
                    val_r += bgr[idx + 2] * kernel[kRow * (2 * r + 1) + kCol];
                    num_pix++;
                }
            }
        }

        int idx = (Row * width + Col) * CHANNEL;
        blur[idx] = val_b / num_pix;
        blur[idx + 1] = val_g / num_pix;
        blur[idx + 2] = val_r / num_pix;
    }
}

int main() {
    // Read the image file
    cv::Mat img = cv::imread("the_starry_night.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return 1;
    }

    // Create a vector of 3 Mat objects
    std::vector<unsigned char> bgr;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            bgr.push_back(img.at<cv::Vec3b>(i, j)[0]);
            bgr.push_back(img.at<cv::Vec3b>(i, j)[1]);
            bgr.push_back(img.at<cv::Vec3b>(i, j)[2]);
        }
    }

    // Create a gaussian convolutional kernel
    int r = 3;
    float sigma = 20.0;
    Eigen::MatrixXf kernel(2 * r + 1, 2 * r + 1);
    for (int i = 0; i < 2 * r + 1; ++i) {
        for (int j = 0; j < 2 * r + 1; ++j) {
            kernel(i, j) = exp(-((i - r) * (i - r) + (j - r) * (j - r)) / (2 * sigma * sigma));
        }
    }

    // Create customised kernel
    // int r = 3;
    // Eigen::MatrixXf kernel(2 * r + 1, 2 * r + 1);
    // kernel <<   1, 0, 0, 1, 0, 0, 1,
    //             0, 1, 0, 1, 0, 1, 0,
    //             0, 0, 1, 1, 1, 0, 0,
    //             0, 0, 0, 1, 0, 0, 0,
    //             0, 0, 1, 1, 1, 0, 0,
    //             0, 1, 0, 1, 0, 1; 0,
    //             1, 0, 0, 1, 0, 0, 1;

    // Allocate memory and move data to GPU
    unsigned char *blur, *bgr_d, *blur_d;
    float *kernel_d;
    int height = img.rows;
    int width = img.cols;
    int size = height * width * sizeof(unsigned char);
    blur = (unsigned char*)malloc(size * CHANNEL);
    cudaMalloc((void**)&blur_d, size * CHANNEL);
    cudaMalloc((void**)&bgr_d, size * CHANNEL);
    cudaMalloc((void**)&kernel_d, (2 * r + 1) * (2 * r + 1) * sizeof(float));
    cudaMemcpy(bgr_d, bgr.data(), size * CHANNEL, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel.data(), (2 * r + 1) * (2 * r + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 grid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 block(16, 16, 1);
    convKernel<<<grid, block>>>(blur_d, bgr_d, kernel_d, r, height, width);
    cudaMemcpy(blur, blur_d, size * CHANNEL, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(blur_d);
    cudaFree(bgr_d);
    cudaFree(kernel_d);

    // Save the blured image
    cv::Mat blur_img(height, width, CV_8UC3, blur);
    cv::imshow("Display window", blur_img);
    cv::waitKey(0); // Wait for a keystroke in the window

    free(blur);

    return 0;
}