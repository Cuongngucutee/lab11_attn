
#include <torch/torch.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h> 
#include <ATen/hip/HIPContext.h> 
#include <vector>
#include <cmath>
#include <cfloat>

using torch::Tensor;

#define CHECK_HIP(x) do { \
    hipError_t err = (x); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

#define WARP_SIZE 64

// -------------------------------------------------------------------------
// Helper: Warp Reduce Sum (Cơ bản & Nhanh nhất trên GPU)
// -------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float val) {
    // Cộng dồn 64 thread trong 1 warp mà không cần __syncthreads
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

// -------------------------------------------------------------------------
// Kernel V40: V38 Logic + Fast Reduction
// -------------------------------------------------------------------------
__global__ void paged_attention_kernel_v40(
    const hip_bfloat16* __restrict__ query,       
    const hip_bfloat16* __restrict__ key_cache,   
    const hip_bfloat16* __restrict__ value_cache, 
    const int* __restrict__ block_tables,         
    const int* __restrict__ seq_indices,          
    const int* __restrict__ kv_lens,              
    const int* __restrict__ q_offsets,            
    const int* __restrict__ query_lens,           
    hip_bfloat16* __restrict__ output,            
    
    const int num_kv_heads,
    const int num_q_heads,
    const int block_size,
    const float scale,
    const float soft_cap,
    
    const int total_tokens_global,
    const int max_num_seqs,
    const int max_blocks_per_seq,
    const int total_blocks_in_cache,

    const long long q_s0, const long long q_s1, const long long q_s2,
    const long long k_s0, const long long k_s1, const long long k_s2, const long long k_s3,
    const long long v_s0, const long long v_s1, const long long v_s2, const long long v_s3,
    const long long o_s0, const long long o_s1, const long long o_s2,
    const long long bt_s0, const long long bt_s1
) {
    // Shared memory nhỏ gọn để giao tiếp giữa các Warp
    __shared__ float s_mem[32]; // Max 32 warps (đủ cho block 2048)

    const int global_token_idx = blockIdx.x; 
    const int q_head_idx       = blockIdx.y; 
    const int tid              = threadIdx.x;
    
    // Warp Info
    const int lane = tid % WARP_SIZE;
    const int wid  = tid / WARP_SIZE;

    // Safety Checks (Giữ nguyên từ V38)
    if (global_token_idx >= total_tokens_global) return;

    const int seq_idx = seq_indices[global_token_idx];
    if (seq_idx < 0 || seq_idx >= max_num_seqs) return;

    const int kv_len  = kv_lens[seq_idx];
    const int q_len   = query_lens[seq_idx];
    const int q_pos   = q_offsets[global_token_idx];

    const int ratio = num_q_heads / num_kv_heads;
    const int kv_head_idx = q_head_idx / ratio;

    long long q_offset = global_token_idx * q_s0 + q_head_idx * q_s1 + tid * q_s2;
    float q_val = static_cast<float>(query[q_offset]);

    float m_prev = -FLT_MAX; 
    float d_prev = 0.0f;
    float acc    = 0.0f;

    const int num_logical_blocks = (kv_len + block_size - 1) / block_size;
    const int causal_limit = kv_len - q_len + q_pos;

    // Block Loop
    for (int b = 0; b < num_logical_blocks; ++b) {
        if (b >= max_blocks_per_seq) break; 

        long long bt_addr = (long long)seq_idx * bt_s0 + (long long)b * bt_s1;
        const int physical_block_idx = block_tables[bt_addr];
        
        if (physical_block_idx < 0 || physical_block_idx >= total_blocks_in_cache) continue;

        int valid_tokens = block_size;
        if (b == num_logical_blocks - 1) {
            valid_tokens = kv_len - b * block_size;
            if (valid_tokens < 0) valid_tokens = 0;
            if (valid_tokens > block_size) valid_tokens = block_size;
        }

        long long k_base = (long long)physical_block_idx * k_s0;
        long long v_base = (long long)physical_block_idx * v_s0;

        // Token Loop
        for (int t = 0; t < valid_tokens; ++t) {
            int kv_abs_pos = b * block_size + t;
            if (kv_abs_pos > causal_limit) continue;

            // A. Dot Product
            long long k_idx = k_base + t * k_s1 + kv_head_idx * k_s2 + tid * k_s3;
            float k_val_load = static_cast<float>(key_cache[k_idx]);
            float dot_elem = q_val * k_val_load;

            // B. Fast Reduction (Warp Shuffle) - Thay thế vòng lặp for chậm chạp
            // 1. Mỗi warp tự tính tổng
            float sum = warpReduceSum(dot_elem);

            // 2. Warp trưởng ghi vào shared memory
            if (lane == 0) s_mem[wid] = sum;
            __syncthreads(); // Sync 1: Đợi các warp ghi xong

            // 3. Warp 0 tính tổng cuối cùng từ shared memory
            float score = (tid < (blockDim.x / WARP_SIZE)) ? s_mem[tid] : 0.0f;
            if (wid == 0) {
                score = warpReduceSum(score);
            }
            
            // Broadcast score cho mọi thread (thông qua shared mem[0])
            if (tid == 0) s_mem[0] = score;
            __syncthreads(); // Sync 2: Đợi warp 0 ghi xong
            score = s_mem[0];

            // C. Online Softmax
            score *= scale;
            if (soft_cap > 0.0f) score = soft_cap * tanhf(score / soft_cap);

            float m_curr = fmaxf(m_prev, score);
            float alpha = expf(score - m_curr);
            float beta  = expf(m_prev - m_curr);

            m_prev = m_curr;
            d_prev = d_prev * beta + alpha;

            // D. Accumulate V
            long long v_idx = v_base + t * v_s1 + kv_head_idx * v_s2 + tid * v_s3;
            float v_val_load = static_cast<float>(value_cache[v_idx]);
            
            acc = acc * beta + alpha * v_val_load;

            // --- IMPORTANT SYNC ---
            // Vẫn cần sync cuối vòng lặp để tránh Race Condition (Batch 32)
            // Nhưng tổng số sync giảm từ 10 xuống 3 -> Tốc độ tăng đáng kể
            __syncthreads();
        }
    }

    // Output
    float res = 0.0f;
    if (d_prev > 1e-10f) res = acc / d_prev;

    long long o_offset = global_token_idx * o_s0 + q_head_idx * o_s1 + tid * o_s2;
    output[o_offset] = static_cast<hip_bfloat16>(res);
}

// -------------------------------------------------------------------------
// Host Code (Giữ nguyên V38 để đảm bảo timing đúng)
// -------------------------------------------------------------------------
Tensor custom_paged_attn(
    const Tensor& query,          
    const Tensor& key_cache,      
    const Tensor& value_cache,
    const std::vector<int64_t>& query_lens,
    const std::vector<int64_t>& kv_lens,
    Tensor block_tables,
    double scale,
    double soft_cap = -1.0 
) {
    auto device = query.device();
    const int64_t num_seqs = query_lens.size();
    
    // Aux Data
    std::vector<int> cpu_seq_indices;
    std::vector<int> cpu_q_offsets;
    int total_tokens = 0;
    
    for(int i=0; i<num_seqs; ++i) {
        int len = query_lens[i];
        for(int j=0; j<len; ++j) {
            cpu_seq_indices.push_back(i);
            cpu_q_offsets.push_back(j);
        }
        total_tokens += len;
    }
    
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(device);
    Tensor seq_indices = torch::empty({total_tokens}, options_int);
    Tensor q_offsets   = torch::empty({total_tokens}, options_int);
    Tensor kv_lens_gpu = torch::empty({(long)kv_lens.size()}, options_int);
    Tensor q_lens_gpu  = torch::empty({(long)query_lens.size()}, options_int);

    CHECK_HIP(hipMemcpy(seq_indices.data_ptr(), cpu_seq_indices.data(), total_tokens * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(q_offsets.data_ptr(), cpu_q_offsets.data(), total_tokens * sizeof(int), hipMemcpyHostToDevice));
    
    std::vector<int> kv_lens_int(kv_lens.begin(), kv_lens.end());
    std::vector<int> q_lens_int(query_lens.begin(), query_lens.end());
    CHECK_HIP(hipMemcpy(kv_lens_gpu.data_ptr(), kv_lens_int.data(), kv_lens_int.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(q_lens_gpu.data_ptr(), q_lens_int.data(), q_lens_int.size() * sizeof(int), hipMemcpyHostToDevice));

    Tensor output = torch::empty({query.size(0), query.size(1), query.size(2)}, query.options());

    const int num_q_heads  = query.size(1);
    const int head_size    = query.size(2);
    const int block_size   = key_cache.size(1);
    const int num_kv_heads = key_cache.size(2);
    const int total_blocks_in_cache = key_cache.size(0);
    const int max_blocks_per_seq = block_tables.size(1);

    dim3 grid(total_tokens, num_q_heads);
    dim3 block(head_size); 

    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    paged_attention_kernel_v40<<<grid, block, 0, stream>>>(
        (const hip_bfloat16*)query.data_ptr(),
        (const hip_bfloat16*)key_cache.data_ptr(),
        (const hip_bfloat16*)value_cache.data_ptr(),
        block_tables.data_ptr<int>(),
        seq_indices.data_ptr<int>(),
        kv_lens_gpu.data_ptr<int>(),
        q_offsets.data_ptr<int>(),
        q_lens_gpu.data_ptr<int>(),
        (hip_bfloat16*)output.data_ptr(),
        
        num_kv_heads, num_q_heads, block_size, (float)scale, (float)soft_cap,
        total_tokens, (int)num_seqs, max_blocks_per_seq, total_blocks_in_cache,

        query.stride(0), query.stride(1), query.stride(2),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        block_tables.stride(0), block_tables.stride(1)
    );
    
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize()); // Giữ lại để đo thời gian đúng
    
    return output;
}

PYBIND11_MODULE(custom_attn, m) {
    m.def("custom_paged_attn", &custom_paged_attn, "Custom Paged Attention V40 (Fastest)");
}
