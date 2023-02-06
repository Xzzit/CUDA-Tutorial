#include <iostream>
#include <math.h>
#include <chrono>
using namespace std::chrono;

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<25; // 1M elements

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }

    // Get starting timepoint
    auto start = high_resolution_clock::now();

    // Run kernel on 1M elements on the CPU
    add(N, x, y);

    // Get ending timepoint
    auto stop = high_resolution_clock::now();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by function: "
    << duration.count() << " microseconds" << std::endl;

    return 0;
}