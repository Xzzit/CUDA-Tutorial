#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define CHANNEL 3

__global__ void colortoGrayscale(
    unsigned char *g, unsigned char *bgr,
    int height, int width) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < width && j < height) {
        int idx_g = j * width + i;
        int idx_bgr = idx_g * CHANNEL;

        unsigned char blue = bgr[idx_bgr];
        unsigned char green = bgr[idx_bgr + 1];
        unsigned char red = bgr[idx_bgr + 2];

        g[idx_g] = 0.21f * red + 0.71f * green + 0.07f * blue;
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
    unsigned char *g, *bgr_d, *g_d;
    int height = img.rows;
    int width = img.cols;
    int size = height * width * sizeof(unsigned char);
    g = (unsigned char*)malloc(size);
    cudaMalloc((void**)&g_d, size);
    cudaMalloc((void**)&bgr_d, size * CHANNEL);
    cudaMemcpy(bgr_d, bgr.data(), size * CHANNEL, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 grid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 block(16, 16, 1);
    colortoGrayscale<<<grid, block>>>(g_d, bgr_d, height, width);
    cudaMemcpy(g, g_d, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(g_d);
    cudaFree(bgr_d);

    // Save the grayscale image
    cv::Mat gray_img(height, width, CV_8UC1, g);
    cv::imwrite("the_starry_night_gray.jpg", gray_img);
    free(g);

    cv::imshow("Display window", img);
    cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}