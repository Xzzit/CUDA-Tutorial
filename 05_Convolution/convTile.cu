#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

#define CHANNEL 3
#define FILTER_RADIUS 6
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2 * FILTER_RADIUS)

__constant__ float kernel[(2*FILTER_RADIUS+1)][(2*FILTER_RADIUS+1)];

__global__ void convKernel(
    unsigned char *blur, unsigned char *bgr,
    int height, int width) {

    // Shared memory
    __shared__ unsigned char bgr_tile[TILE_SIZE][TILE_SIZE][CHANNEL];

    // Indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * blockDim.x + tx;
    int Row = by * blockDim.y + ty;

    if (Col >= width || Row >= height) return;

    int num_pix = 0;
    int val_b = 0;
    int val_g = 0;
    int val_r = 0;

    for (int i = 0; i < ceil(float(TILE_SIZE)/BLOCK_SIZE); i++) {
        for (int j = 0; j < ceil(float(TILE_SIZE)/BLOCK_SIZE); j++) {

            int inRow = Row - FILTER_RADIUS + i * BLOCK_SIZE;
            int inCol = Col - FILTER_RADIUS + j * BLOCK_SIZE;

            // Load the tile into shared memory
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                for (int k = 0; k < CHANNEL; k++) {
                    bgr_tile[ty+i*BLOCK_SIZE][tx+j*BLOCK_SIZE][k] = bgr[(inRow * width + inCol) * CHANNEL + k];
                }
            }
            else {
                for (int k = 0; k < CHANNEL; k++) {
                    bgr_tile[ty+i*BLOCK_SIZE][tx+j*BLOCK_SIZE][k] = 0;
                }
            }
            __syncthreads();
        }
    }

    // Convolution
    for (int kRow = 0; kRow < 2 * FILTER_RADIUS + 1; kRow++) {
        for (int kCol = 0; kCol < 2 * FILTER_RADIUS + 1; kCol++) {
            val_b += bgr_tile[ty + kRow][tx + kCol][0] * kernel[kRow][kCol];
            val_g += bgr_tile[ty + kRow][tx + kCol][1] * kernel[kRow][kCol];
            val_r += bgr_tile[ty + kRow][tx + kCol][2] * kernel[kRow][kCol];
            num_pix++;
            __syncthreads();
        }
    }

    int idx = (Row * width + Col) * CHANNEL;
    blur[idx] = val_b / num_pix;
    blur[idx + 1] = val_g / num_pix;
    blur[idx + 2] = val_r / num_pix;
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
    float kernel_h[(2*FILTER_RADIUS+1)][(2*FILTER_RADIUS+1)];
    int r = FILTER_RADIUS;
    float sigma = 10.0;
    for (int i = 0; i < 2 * r + 1; ++i) {
        for (int j = 0; j < 2 * r + 1; ++j) {
            kernel_h[i][j] = exp(-((i - r) * (i - r) + (j - r) * (j - r)) / (2 * sigma * sigma));
        }
    }

    // Allocate memory and move data to GPU
    // Note that we don't need to allocate memory for __constant__ variables
    unsigned char *blur, *bgr_d, *blur_d;
    int height = img.rows;
    int width = img.cols;
    int size = height * width * sizeof(unsigned char);
    blur = (unsigned char*)malloc(size * CHANNEL);
    cudaMalloc((void**)&blur_d, size * CHANNEL);
    cudaMalloc((void**)&bgr_d, size * CHANNEL);
    cudaMemcpy(bgr_d, bgr.data(), size * CHANNEL, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel, &kernel_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));

    // Kernel launch
    dim3 grid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    convKernel<<<grid, block>>>(blur_d, bgr_d, height, width);
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