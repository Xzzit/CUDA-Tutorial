#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define CHANNEL 3
#define KERNELSIZE 3

__global__ void blurKernel(
    unsigned char *blur, unsigned char *bgr,
    int height, int width) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < width && j < height) {
        int num_pix = 0;
        int val_b = 0;
        int val_g = 0;
        int val_r = 0;

        for (int col = -KERNELSIZE; col <= KERNELSIZE; col++) {
            for (int row = -KERNELSIZE; row <= KERNELSIZE; row++) {
                int idx = ((j + row) * width + (i + col)) * CHANNEL;
                if (idx >= 0 && idx < height * width * CHANNEL) {
                    val_b += bgr[idx];
                    val_g += bgr[idx + 1];
                    val_r += bgr[idx + 2];
                    num_pix++;
                }
            }
        }

        int idx = (j * width + i) * CHANNEL;
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

    // Allocate memory and move data to GPU
    unsigned char *blur, *bgr_d, *blur_d;
    int height = img.rows;
    int width = img.cols;
    int size = height * width * sizeof(unsigned char);
    blur = (unsigned char*)malloc(size * CHANNEL);
    cudaMalloc((void**)&blur_d, size * CHANNEL);
    cudaMalloc((void**)&bgr_d, size * CHANNEL);
    cudaMemcpy(bgr_d, bgr.data(), size * CHANNEL, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 grid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 block(16, 16, 1);
    blurKernel<<<grid, block>>>(blur_d, bgr_d, height, width);
    cudaMemcpy(blur, blur_d, size * CHANNEL, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(blur_d);
    cudaFree(bgr_d);

    // Save the blured image
    cv::Mat blur_img(height, width, CV_8UC3, blur);
    cv::imshow("Display window", blur_img);
    cv::waitKey(0); // Wait for a keystroke in the window

    free(blur);

    return 0;
}