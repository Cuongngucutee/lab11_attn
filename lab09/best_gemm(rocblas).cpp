#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h> 

__global__ void verify_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) { hipError_t error = cmd; if (error != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(error) << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

#ifndef CHECK_ROCBLAS
#define CHECK_ROCBLAS(cmd) { rocblas_status status = cmd; if (status != rocblas_status_success) { std::cerr << "rocBLAS Error: " << status << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

using namespace std;

void compare_results(const float* C_rocblas, const float* C_naive, int M, int N) {
    std::cout << "Verifying... Comparing rocBLAS result vs Naive GPU result..." << std::endl;
    
    int errors = 0;
    float epsilon = 1e-1;

    for (size_t i = 0; i < (size_t)M * N; ++i) {
        float diff = std::abs(C_rocblas[i] - C_naive[i]);
        if (diff > epsilon) {
            if (errors < 5) {
                std::cerr << "Mismatch at index " << i
                          << ": rocBLAS=" << C_rocblas[i]
                          << ", Naive=" << C_naive[i]
                          << ", Diff=" << diff << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "Result Verification: CORRECT (Checked 100% of " << M * N << " elements)" << std::endl;
    } else {
        std::cout << "Result Verification: FAILED with " << errors << " mismatches." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    for (int i = 1; i < argc; i++) {
        std::string option(argv[i]);
        if (i + 1 < argc) {
            if (option == "-m") M = std::atoi(argv[++i]);
            else if (option == "-n") N = std::atoi(argv[++i]);
            else if (option == "-k") K = std::atoi(argv[++i]);
        }
    }

    std::cout << "Best GEMM (rocBLAS + GPU Cross-Check) M=" << M << " N=" << N << " K=" << K << std::endl;

    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_rocblas(size_C);

    srand(time(NULL));
    for (size_t i = 0; i < size_A; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < size_B; i++) h_B[i] = (float)(rand() % 10) / 10.0f;

    float *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_HIP(hipMalloc(&d_A, size_A * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_B, size_B * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C, size_C * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C_ref, size_C * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice));

    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_ROCBLAS(rocblas_sgemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    ));
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    int n_iter = 20;
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < n_iter; i++) {
        rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N
        );
    }
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
    double gflops = (2.0 * M * N * K) / ((milliseconds / n_iter) * 1e6);

    std::cout << "Time (rocBLAS avg): " << milliseconds / n_iter << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    verify_kernel_naive<<<grid, block>>>(d_A, d_B, d_C_ref, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_C_rocblas.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> h_C_naive(size_C);
    CHECK_HIP(hipMemcpy(h_C_naive.data(), d_C_ref, size_C * sizeof(float), hipMemcpyDeviceToHost));

    compare_results(h_C_rocblas.data(), h_C_naive.data(), M, N);

    CHECK_ROCBLAS(rocblas_destroy_handle(handle));
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));
    CHECK_HIP(hipFree(d_C_ref));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    return 0;
}
