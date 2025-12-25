#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <hip/hip_runtime.h>

#define WAVE_SIZE 64
#define BM 64   
#define BN 64  
#define BK 16 
#define TM 4   
#define TN 4    
#define NUM_THREADS 256 

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) { hipError_t error = cmd; if (error != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(error) << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

using namespace std;

__global__ __launch_bounds__(NUM_THREADS)
void warp_tiled_v3_fixed(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    
    const int tid = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE; 
    const int lane_id = tid % WAVE_SIZE; 
    const int wave_row_offset = (wave_id / 2) * 32; 
    const int wave_col_offset = (wave_id % 2) * 32;

    __shared__ float As[BK][BM]; 
    __shared__ float Bs[BK][BN]; 

    float thread_results[TM * TN] = {0.0f};


    const float* A_ptr = A + blockIdx.y * BM * K;
    const float* B_ptr = B + blockIdx.x * BN;
    float* C_ptr = C;

    for (int k_loop = 0; k_loop < K; k_loop += BK) {
        {

            int a_vec_col = tid % 4;       
            int a_row     = tid / 4;     
            
            float4 tmp_a = reinterpret_cast<const float4*>(&A_ptr[a_row * K + (a_vec_col * 4)])[0];
            
            As[a_vec_col * 4 + 0][a_row] = tmp_a.x;
            As[a_vec_col * 4 + 1][a_row] = tmp_a.y;
            As[a_vec_col * 4 + 2][a_row] = tmp_a.z;
            As[a_vec_col * 4 + 3][a_row] = tmp_a.w;
        }
        {
            int b_vec_col = tid % 16;       
            int b_row     = tid / 16;       
            
            float4 tmp_b = reinterpret_cast<const float4*>(&B_ptr[b_row * N + (b_vec_col * 4)])[0];
            
            reinterpret_cast<float4*>(&Bs[b_row][b_vec_col * 4])[0] = tmp_b;
        }
        
        __syncthreads();
        A_ptr += BK;
        B_ptr += BK * N;

        const int thread_row_in_wave = (lane_id / 8) * TM; 
        const int thread_col_in_wave = (lane_id % 8) * TN; 

        for (int dot = 0; dot < BK; ++dot) {
            float reg_M[TM];
            float reg_N[TN];
            #pragma unroll
            for (int i=0; i<TM; ++i) {
                reg_M[i] = As[dot][wave_row_offset + thread_row_in_wave + i];
            }
            
            #pragma unroll
            for (int i=0; i<TN; ++i) {
                reg_N[i] = Bs[dot][wave_col_offset + thread_col_in_wave + i];
            }


            #pragma unroll
            for (int i=0; i<TM; ++i) {
                #pragma unroll
                for (int j=0; j<TN; ++j) {
                    thread_results[i * TN + j] += reg_M[i] * reg_N[j];
                }
            }
        }
        __syncthreads();
    }


    const int thread_row_in_wave = (lane_id / 8) * TM;
    const int thread_col_in_wave = (lane_id % 8) * TN;
    
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = blockIdx.y * BM + wave_row_offset + thread_row_in_wave + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int c = blockIdx.x * BN + wave_col_offset + thread_col_in_wave + j;
            if (r < M && c < N) {
                float4 res;
                res.x = thread_results[i * TN + j + 0];
                res.y = thread_results[i * TN + j + 1];
                res.z = thread_results[i * TN + j + 2];
                res.w = thread_results[i * TN + j + 3];
                reinterpret_cast<float4*>(&C_ptr[r * N + c])[0] = res;
            }
        }
    }
}

__global__ void verify_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void compare_results(const float* C_opt, const float* C_ref, int M, int N) {
    std::cout << "Verifying... Comparing Warp V3 vs Naive GPU..." << std::endl;
    long long errors = 0; float epsilon = 1e-1;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        if (std::abs(C_opt[i] - C_ref[i]) > epsilon) {
            if (errors < 5) std::cerr << "Mismatch idx " << i << ": Opt=" << C_opt[i] << ", Ref=" << C_ref[i] << std::endl;
            errors++;
        }
    }
    if (errors == 0) std::cout << "Result Verification: CORRECT" << std::endl;
    else std::cout << "Result Verification: FAILED (" << errors << " errors)" << std::endl;
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
    std::cout << "Warp Tiled V3 (Fixed Logic) M=" << M << " N=" << N << " K=" << K << std::endl;

    size_t size_A = (size_t)M * K; size_t size_B = (size_t)K * N; size_t size_C = (size_t)M * N;
    vector<float> h_A(size_A), h_B(size_B);
    srand(time(NULL));
    for (size_t i = 0; i < size_A; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < size_B; i++) h_B[i] = (float)(rand() % 10) / 10.0f;

    float *d_A, *d_B, *d_C_opt, *d_C_ref;
    CHECK_HIP(hipMalloc(&d_A, size_A * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_B, size_B * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C_opt, size_C * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_C_ref, size_C * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice));


    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    warp_tiled_v3_fixed<<<grid, block>>>(d_A, d_B, d_C_opt, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start)); CHECK_HIP(hipEventCreate(&stop));

    int n_iter = 20;
    CHECK_HIP(hipEventRecord(start));
    for(int i=0; i<n_iter; i++) warp_tiled_v3_fixed<<<grid, block>>>(d_A, d_B, d_C_opt, M, N, K);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
    double gflops = (2.0 * M * N * K) / ((milliseconds/n_iter) * 1e6);

    std::cout << "Time (Warp V3 avg): " << milliseconds/n_iter << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

 
    dim3 br(32, 32); dim3 gr((N + 31) / 32, (M + 31) / 32);
    verify_kernel_naive<<<gr, br>>>(d_A, d_B, d_C_ref, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    vector<float> h_C_opt(size_C), h_C_ref(size_C);
    CHECK_HIP(hipMemcpy(h_C_opt.data(), d_C_opt, size_C * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_C_ref.data(), d_C_ref, size_C * sizeof(float), hipMemcpyDeviceToHost));
    compare_results(h_C_opt.data(), h_C_ref.data(), M, N);

    CHECK_HIP(hipFree(d_A)); CHECK_HIP(hipFree(d_B)); CHECK_HIP(hipFree(d_C_opt)); CHECK_HIP(hipFree(d_C_ref));
    return 0;
}