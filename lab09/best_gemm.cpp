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
#define NUM_THREADS 256 

#ifndef CHECK_HIP
#define CHECK_HIP(cmd) { hipError_t error = cmd; if (error != hipSuccess) { std::cerr << "HIP Error: " << hipGetErrorString(error) << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#endif

using namespace std;


typedef float v4f __attribute__((ext_vector_type(4)));

__device__ inline v4f mfma_16x16x1(float a, float b, v4f c) {
    return __builtin_amdgcn_mfma_f32_16x16x1f32(a, b, c, 0, 0, 0);
}


__global__ __launch_bounds__(256)
void mysgemm_v6_mfma_16x16(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    int tid = threadIdx.x;
    int wid = tid / WAVE_SIZE; 
    int lane = tid % WAVE_SIZE; 


    int wave_row_off = (wid / 2) * 32; 
    int wave_col_off = (wid % 2) * 32;

    int global_row_base = blockIdx.y * BM + wave_row_off;
    int global_col_base = blockIdx.x * BN + wave_col_off;

    __shared__ float As[BK][BM]; 
    __shared__ float Bs[BK][BN];

    v4f c00 = {0}, c01 = {0}, c10 = {0}, c11 = {0};

    const float* A_ptr = A + blockIdx.y * BM * K;
    const float* B_ptr = B + blockIdx.x * BN;
    float* C_ptr = C;

    for (int k = 0; k < K; k += BK) {
        
        for (int i = 0; i < 4; ++i) {
             int load_idx = tid + i * 256;
             int r = load_idx / BK; 
             int c = load_idx % BK;
             if (r < BM) As[c][r] = A_ptr[r * K + c];
        }


        for (int i = 0; i < 4; ++i) {
            int load_idx = tid + i * 256;
            int r = load_idx / BN; 
            int c = load_idx % BN;
            if (r < BK) Bs[r][c] = B_ptr[r * N + c];
        }

        __syncthreads();
        A_ptr += BK; B_ptr += BK * N;

        for (int dot = 0; dot < BK; ++dot) {
            
            float a_val_0 = As[dot][wave_row_off + (lane % 16)];     
            float a_val_1 = As[dot][wave_row_off + (lane % 16) + 16]; 
            
            float b_val_0 = Bs[dot][wave_col_off + (lane % 16)];      
            float b_val_1 = Bs[dot][wave_col_off + (lane % 16) + 16]; 

            c00 = mfma_16x16x1(a_val_0, b_val_0, c00); 
   
            c01 = mfma_16x16x1(a_val_0, b_val_1, c01); 
      
            c10 = mfma_16x16x1(a_val_1, b_val_0, c10); 
           
            c11 = mfma_16x16x1(a_val_1, b_val_1, c11); 
        }
        __syncthreads();
    }
    
    auto store_block = [&](v4f val, int r_off, int c_off) {
        for (int i = 0; i < 4; ++i) {
            int r_local = (lane / 16) + 4 * i; 
            int c_local = lane % 16;           
            
            int r = global_row_base + r_off + r_local;
            int c = global_col_base + c_off + c_local;
            
            if (r < M && c < N) C_ptr[r * N + c] = val[i];
        }
    };

    store_block(c00, 0, 0);
    store_block(c01, 0, 16);
    store_block(c10, 16, 0);
    store_block(c11, 16, 16);
}

// --- VERIFICATION NAIVE ---
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
    std::cout << "Verifying... Comparing MFMA V6 vs Naive GPU..." << std::endl;
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
    std::cout << "Best GEMM (V6 FIX: MFMA 16x16x1f32) M=" << M << " N=" << N << " K=" << K << std::endl;

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

    dim3 block(256); 
    dim3 grid((N + 63) / 64, (M + 63) / 64);

    mysgemm_v6_mfma_16x16<<<grid, block>>>(d_A, d_B, d_C_opt, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start)); CHECK_HIP(hipEventCreate(&stop));

    int n_iter = 20;
    CHECK_HIP(hipEventRecord(start));
    for(int i=0; i<n_iter; i++) mysgemm_v6_mfma_16x16<<<grid, block>>>(d_A, d_B, d_C_opt, M, N, K);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
    double gflops = (2.0 * M * N * K) / ((milliseconds/n_iter) * 1e6);

    std::cout << "Time (V6 MFMA): " << milliseconds/n_iter << " ms" << std::endl;
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