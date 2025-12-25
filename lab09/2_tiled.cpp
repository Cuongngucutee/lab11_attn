#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <hip/hip_runtime.h>

#define TILE_WIDTH 32

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) { hipError_t error = cmd; if (error != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(error) << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

using namespace std;

__global__ void mysgemm_tiled_improved(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    float Cvalue = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        if (Row < M && (m * TILE_WIDTH + tx) < K) {
            As[ty][tx] = A[Row * K + (m * TILE_WIDTH + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((m * TILE_WIDTH + ty) < K && Col < N) {
            Bs[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (Row < M && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

void verify_result(const float* A, const float* B, const float* C, int M, int N, int K) {
    if (M > 2048) { cout << "Skipping CPU verify for large matrix." << endl; return; }
    int errors = 0; float epsilon = 1e-1;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            if (abs(C[i * N + j] - sum) > epsilon) { if(errors<5) cerr<<"Err "<<i<<","<<j<<endl; errors++; }
        }
    }
    if(errors==0) cout<<"Result Verification: CORRECT"<<endl;
    else cout<<"Result Verification: FAILED"<<endl;
}

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    for (int i = 1; i < argc; i++) {
        string option(argv[i]);
        if (i+1 < argc) {
            if (option == "-m") M = stoi(argv[++i]);
            else if (option == "-n") N = stoi(argv[++i]);
            else if (option == "-k") K = stoi(argv[++i]);
        }
    }
    cout << "Tiled GEMM (Improved Block Level) M=" << M << " N=" << N << " K=" << K << endl;

    size_t size_A = (size_t)M * K; size_t size_B = (size_t)K * N; size_t size_C = (size_t)M * N;
    vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    srand(42);
    for(size_t i=0; i<size_A; i++) h_A[i] = (float)(rand()%10)/10.0f;
    for(size_t i=0; i<size_B; i++) h_B[i] = (float)(rand()%10)/10.0f;

    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, size_A * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_B, size_B * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C, size_C * sizeof(float)));
    
    CHECK_HIP(hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    mysgemm_tiled_improved<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start)); CHECK_HIP(hipEventCreate(&stop));
    
    CHECK_HIP(hipEventRecord(start));
    mysgemm_tiled_improved<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
    cout << "Time taken: " << milliseconds << " ms" << endl;
    cout << "Performance: " << (2.0*M*N*K)/(milliseconds*1e6) << " GFLOP/s" << endl;

    CHECK_HIP(hipMemcpy(h_C.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));
    verify_result(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    CHECK_HIP(hipFree(d_A)); CHECK_HIP(hipFree(d_B)); CHECK_HIP(hipFree(d_C));
    return 0;
}