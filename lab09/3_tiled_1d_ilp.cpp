#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <hip/hip_runtime.h>
#define BM 64
#define BN 64
#define BK 16
#define TM 8
#define NUM_THREADS 512
#define PADDING 4
#ifndef CHECK_HIP
#define CHECK_HIP(cmd) { hipError_t error = cmd; if (error != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(error) << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

using namespace std;

__global__ __launch_bounds__(512)
void mysgemm_1d_ilp_improved(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;

    const int tx = tid % BN;
    const int ty = tid / BN;

    __shared__ float As[BK][BM + PADDING];

    __shared__ float Bs[BK][BN + PADDING];

    float thread_results[TM] = {0.0f};
    float reg_A[TM];


    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;
    float* C_ptr = C + by * BM * N + bx * BN;

    for (int k = 0; k < K; k += BK) {
        
        int load_a_idx1 = tid; 
        int row_a1 = load_a_idx1 / BK; 
        int col_a1 = load_a_idx1 % BK;
        float val_a1 = A_ptr[row_a1 * K + col_a1];
        As[col_a1][row_a1] = val_a1;

        int load_a_idx2 = tid + 512;
        if (load_a_idx2 < BM * BK) { 
             int row_a2 = load_a_idx2 / BK;
             int col_a2 = load_a_idx2 % BK;
             float val_a2 = A_ptr[row_a2 * K + col_a2];
             As[col_a2][row_a2] = val_a2;
        }

        int load_b_idx1 = tid;
        int row_b1 = load_b_idx1 / BN;
        int col_b1 = load_b_idx1 % BN;
        float val_b1 = B_ptr[row_b1 * N + col_b1];
        Bs[row_b1][col_b1] = val_b1;

        int load_b_idx2 = tid + 512;
        if (load_b_idx2 < BK * BN) {
            int row_b2 = load_b_idx2 / BN;
            int col_b2 = load_b_idx2 % BN;
            float val_b2 = B_ptr[row_b2 * N + col_b2];
            Bs[row_b2][col_b2] = val_b2;
        }

        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            float tmp_b = Bs[dot_idx][tx];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = As[dot_idx][ty * TM + i];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                thread_results[i] += reg_A[i] * tmp_b;
            }
        }
        __syncthreads();
    }

    int row_start = ty * TM;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        C_ptr[(row_start + i) * N + tx] = thread_results[i];
    }
}

void verify_result(const float* A, const float* B, const float* C, int M, int N, int K) {
    if (M > 2048) { std::cout << "Skipping verification for large matrix." << std::endl; return; }
    int errors = 0; float epsilon = 1e-1;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            if (std::abs(C[i * N + j] - sum) > epsilon) { if(errors<5) std::cerr << "Err " << i << "," << j << std::endl; errors++; }
        }
    }
    if(errors==0) std::cout<<"Result Verification: CORRECT"<<std::endl;
    else std::cout<<"Result Verification: FAILED"<<std::endl;
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
    std::cout << "1D-ILP Tiled Improved (Transposed Shared Mem) M=" << M << " N=" << N << " K=" << K << std::endl;

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

    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    mysgemm_1d_ilp_improved<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start)); CHECK_HIP(hipEventCreate(&stop));
    
    CHECK_HIP(hipEventRecord(start));
    mysgemm_1d_ilp_improved<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << (2.0*M*N*K)/(milliseconds*1e6) << " GFLOP/s" << std::endl;

    CHECK_HIP(hipMemcpy(h_C.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));
    verify_result(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    CHECK_HIP(hipFree(d_A)); CHECK_HIP(hipFree(d_B)); CHECK_HIP(hipFree(d_C));
    return 0;
}