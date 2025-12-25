#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <hip/hip_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS 256
#define PADDING 4 
#ifndef CHECK_HIP
#define CHECK_HIP(cmd) { hipError_t error = cmd; if (error != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(error) << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

using namespace std;

__global__ __launch_bounds__(256)
void mysgemm_2d_ilp_improved(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;

    const int ty = tid / (BN / TN);
    const int tx = tid % (BN / TN);
    const int tid_a_row = tid / (BK / 4);       
    const int tid_a_col = (tid % (BK / 4)) * 4;

    const int tid_b_row = tid / (BN / 4);
    const int tid_b_col = (tid % (BN / 4)) * 4;


    __shared__ float As[BK][BM + PADDING];
    __shared__ float Bs[BK][BN + PADDING];

    float thread_results[TM * TN] = {0.0f};
    float reg_M[TM];
    float reg_N[TN];

    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;
    float* C_ptr = C + by * BM * N + bx * BN;

    for (int k = 0; k < K; k += BK) {
        
        float4 tmp_a = reinterpret_cast<const float4*>(&A_ptr[tid_a_row * K + tid_a_col])[0];
        
        As[tid_a_col + 0][tid_a_row] = tmp_a.x;
        As[tid_a_col + 1][tid_a_row] = tmp_a.y;
        As[tid_a_col + 2][tid_a_row] = tmp_a.z;
        As[tid_a_col + 3][tid_a_row] = tmp_a.w;

        float4 tmp_b = reinterpret_cast<const float4*>(&B_ptr[tid_b_row * N + tid_b_col])[0];
        
        reinterpret_cast<float4*>(&Bs[tid_b_row][tid_b_col])[0] = tmp_b;

        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_M[i] = As[dot_idx][ty * TM + i];
            }

            #pragma unroll
            for (int i = 0; i < TN; ++i) {
                reg_N[i] = Bs[dot_idx][tx * TN + i];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    thread_results[i * TN + j] += reg_M[i] * reg_N[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int c_row = ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int c_col = tx * TN + j;
            
            float4 res;
            res.x = thread_results[i * TN + j + 0];
            res.y = thread_results[i * TN + j + 1];
            res.z = thread_results[i * TN + j + 2];
            res.w = thread_results[i * TN + j + 3];

            reinterpret_cast<float4*>(&C_ptr[c_row * N + c_col])[0] = res;
        }
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
    std::cout << "2D-ILP Improved (Padding + Transpose) M=" << M << " N=" << N << " K=" << K << std::endl;

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
    
    mysgemm_2d_ilp_improved<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start)); CHECK_HIP(hipEventCreate(&stop));
    
    CHECK_HIP(hipEventRecord(start));
    mysgemm_2d_ilp_improved<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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