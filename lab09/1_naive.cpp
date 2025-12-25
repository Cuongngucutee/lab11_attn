#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <tuple>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

using namespace std;

template<typename T>
__global__ void matmul_kernel(const T *a, const T *b, T *c, int M, int N, int K) {
  int col = blockIdx.x * 32 + (threadIdx.x % 32);
  int row = blockIdx.y * 32 + (threadIdx.x / 32);
  
  if (row < M && col < N) {
    T sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += a[row * K + k] * b[k * N + col]; 
    }
    c[row * N + col] = sum;
  }
}

template<typename T>
__host__ void verifyResult(T *a, T *b, T *c, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += a[i * K + k] * b[k * N + j];
      }
      if (c[i * N + j] != sum) {
          fprintf(stderr, "Mismatch at c[%d][%d]: expected %d, got %d\n", i, j, sum, c[i * N + j]);
          assert(c[i * N + j] == sum);
      }
    }
  }
  cout << "Result is correct!\n";
}

template<typename T>
__host__ void copyFromHostToDevice(T *h_a, T *h_b, T *d_a, T *d_b, int M, int N, int K) {
  size_t a_bytes = M * K * sizeof(T);
  size_t b_bytes = K * N * sizeof(T);
  
  hipError_t err = hipMemcpy(d_a, h_a, a_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_a to d_a (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMemcpy(d_b, h_b, b_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_b to d_b (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void executeKernel(T *d_a, T *d_b, T *d_c, int M, int N, int K, float &milliseconds) {
  int block_dim = 32;
  dim3 block(block_dim * block_dim); 
  dim3 grid((N + block_dim - 1) / block_dim, (M + block_dim - 1) / block_dim);

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start, 0);

  matmul_kernel<T><<<grid, block>>>(d_a, d_b, d_c, M, N, K);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);

  hipEventElapsedTime(&milliseconds, start, stop);
  
  hipEventDestroy(start);
  hipEventDestroy(stop);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to launch kernel (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void copyFromDeviceToHost(T *d_c, T *h_c, int M, int N) {
  size_t bytes = M * N * sizeof(T);
  hipError_t err = hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy d_c to h_c (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void deallocateMemory(T *d_a, T *d_b, T *d_c) {
  hipError_t err = hipFree(d_a);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to free d_a (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  hipFree(d_b);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to free d_b (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  hipFree(d_c);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to free d_c (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void cleanUpDevice() {
  hipError_t err = hipDeviceReset();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to clean up device (error code %s)\n", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024;
  int N = 1024;
  int K = 1024;

  for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    if (i + 1 >= argc) break; 
    std::string value(argv[i+1]);
    
    if (option.compare("-m") == 0) {
      M = std::stoi(value);
      i++;
    }
    else if (option.compare("-n") == 0) {
      N = std::stoi(value);
      i++;
    }
    else if (option.compare("-k") == 0) {
      K = std::stoi(value);
      i++;
    }
  }
  return {M, N, K};
}

int main(int argc, char *argv[]) {
  auto parsedCmdLineArgsTuple = parseCmdLineArgs(argc, argv);
  int M = std::get<0>(parsedCmdLineArgsTuple);
  int N = std::get<1>(parsedCmdLineArgsTuple);
  int K = std::get<2>(parsedCmdLineArgsTuple);
  
  printf("Matrix Size: M=%d, N=%d, K=%d\n", M, N, K);

  int *h_a = (int *)malloc(M * K * sizeof(int));
  int *h_b = (int *)malloc(K * N * sizeof(int));
  int *h_c = (int *)malloc(M * N * sizeof(int));

  srand(42);
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      h_a[i * K + j] = rand() % 10;
    }
  }

  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      h_b[i * N + j] = rand() % 10;
    }
  }
  
  int *d_a, *d_b, *d_c;
  hipMalloc((void **)&d_a, M * K * sizeof(int));
  hipMalloc((void **)&d_b, K * N * sizeof(int));
  hipMalloc((void **)&d_c, M * N * sizeof(int));

  copyFromHostToDevice<int>(h_a, h_b, d_a, d_b, M, N, K);

  float time = 0;
  executeKernel<int>(d_a, d_b, d_c, M, N, K, time);

  printf("Time taken for GEMM: %f ms\n", time);
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = (flops / (time / 1000.0)) / 1e9;
  std::cout << "Performance: " << gflops << " GFLOP/s\n";

  copyFromDeviceToHost<int>(d_c, h_c, M, N);
  
  if (M <= 1024 && N <= 1024) {
      verifyResult<int>(h_a, h_b, h_c, M, N, K);
  } else {
      std::cout << "Skipping CPU verification for large matrices.\n";
  }

  deallocateMemory<int>(d_a, d_b, d_c);
  
  free(h_a); free(h_b); free(h_c);
  return 0;
}