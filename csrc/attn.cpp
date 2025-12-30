#include <torch/torch.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <vector>
#include <cmath>

using torch::Tensor;

#define WARP_SIZE 64

struct alignas(16) Vec128 {
    union {
        float4 f4;
        hip_bfloat16 bf16[8];
    };
};

__device__ __forceinline__ float vec_dot(const Vec128& a, const Vec128& b) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum += static_cast<float>(a.bf16[i]) * static_cast<float>(b.bf16[i]);
    }
    return sum;
}

// --- Main Kernel (Templated Warps) ---
// Added NUM_WARPS as a template parameter so compiler can optimize constants
template<int HEAD_SIZE, int BLOCK_SIZE, int NUM_WARPS>
__global__ void paged_attn_kernel_opt(
    hip_bfloat16* __restrict__ output,
    hip_bfloat16* __restrict__ temp_acc, 
    float* __restrict__ temp_meta,     
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k_cache,
    const hip_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int64_t* __restrict__ kv_lens,
    const int64_t* __restrict__ q_lens,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride_0, const int q_stride_1,
    const int k_stride_block, const int k_stride_token, const int k_stride_head,
    const int v_stride_block, const int v_stride_token, const int v_stride_head,
    const int num_kv_heads,
    const int num_splits,
    const int num_q_heads_real
) {
    const int q_token_global_idx = blockIdx.x; 
    
    // Compile-time constant indexing
    const int head_idx = blockIdx.y * NUM_WARPS + threadIdx.y;
    
    if (head_idx >= num_q_heads_real) return;

    const int split_idx = blockIdx.z;

    // --- 1. Metadata ---
    int local_seq_idx = 0;
    int current_q_start = 0;
    int q_len_val = 0;
    
    // Fast Scan
    if (gridDim.x > 1) {
       int limit = gridDim.x; 
       for(int i=0; i < limit; ++i) {
           int len = (int)q_lens[i];
           if (q_token_global_idx < current_q_start + len) {
               local_seq_idx = i;
               q_len_val = len;
               break;
           }
           current_q_start += len;
       }
    } else {
       q_len_val = (int)q_lens[0];
    }
    
    const int kv_len = (int)kv_lens[local_seq_idx];
    const int q_pos_in_seq = q_token_global_idx - current_q_start;
    const int num_q_heads = num_q_heads_real;
    const int group_size = num_q_heads / num_kv_heads; 
    const int kv_head_idx = head_idx / group_size;

    // --- 2. Split Range ---
    const int total_logical_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (total_logical_blocks + num_splits - 1) / num_splits;
    const int start_blk = split_idx * blocks_per_split;
    const int end_blk = min(start_blk + blocks_per_split, total_logical_blocks);
    
    if (start_blk >= end_blk) {
        if (num_splits > 1 && threadIdx.x == 0 && threadIdx.y == 0) {
            int meta_idx = (q_token_global_idx * num_q_heads * num_splits + head_idx * num_splits + split_idx) * 2;
            temp_meta[meta_idx] = -1e20f;     
            temp_meta[meta_idx + 1] = 0.0f;   
        }
        return;
    }

    // --- 3. Load Q ---
    const int VECS_PER_HEAD = HEAD_SIZE / 8;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    Vec128 q_vec;
    #pragma unroll
    for(int i=0; i<8; ++i) q_vec.bf16[i] = hip_bfloat16(0.0f);

    if (tid < VECS_PER_HEAD) {
        const float4* q_ptr_f4 = reinterpret_cast<const float4*>(
            q + q_token_global_idx * q_stride_0 + head_idx * q_stride_1
        );
        q_vec.f4 = q_ptr_f4[tid];
        #pragma unroll
        for(int i=0; i<8; ++i) {
             float val = static_cast<float>(q_vec.bf16[i]);
             q_vec.bf16[i] = hip_bfloat16(val * scale);
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = static_cast<float>(q_vec.bf16[i]);
        val = __shfl(val, lane_id % VECS_PER_HEAD, WARP_SIZE);
        q_vec.bf16[i] = hip_bfloat16(val);
    }
    
    // --- 4. Main Loop ---
    float m_i = -1e20f; 
    float l_i = 0.0f;   
    float acc[8] = {0.0f};

    constexpr int TOKENS_PER_WARP = WARP_SIZE / VECS_PER_HEAD;
    const int token_idx_in_group = tid / VECS_PER_HEAD; 
    const int vec_idx = tid % VECS_PER_HEAD;
    const int* my_block_table = block_tables + local_seq_idx * max_num_blocks_per_seq;
    const int causal_limit = kv_len - q_len_val + q_pos_in_seq;
    
    int loop_end_safe = end_blk;
    if (end_blk == total_logical_blocks) loop_end_safe = end_blk - 1;

    const long long k_head_offset = (long long)kv_head_idx * k_stride_head;
    const long long v_head_offset = (long long)kv_head_idx * v_stride_head;

    // A. SAFE LOOP
    if (start_blk < loop_end_safe) {
        for (int blk_idx = start_blk; blk_idx < loop_end_safe; ++blk_idx) {
            const int phys_block = my_block_table[blk_idx];
            
            #pragma unroll (HEAD_SIZE >= 256 ? 4 : 16)
            for (int t = 0; t < BLOCK_SIZE; t += TOKENS_PER_WARP) {
                int current_token_off = t + token_idx_in_group;
                long long k_offset = k_head_offset + (long long)phys_block * k_stride_block + (long long)current_token_off * k_stride_token;
                const float4* k_ptr = reinterpret_cast<const float4*>(k_cache + k_offset);
                Vec128 k_vec; k_vec.f4 = k_ptr[vec_idx];

                long long v_offset = v_head_offset + (long long)phys_block * v_stride_block + (long long)current_token_off * v_stride_token;
                const float4* v_ptr = reinterpret_cast<const float4*>(v_cache + v_offset);
                Vec128 v_vec; v_vec.f4 = v_ptr[vec_idx];

                float score = vec_dot(q_vec, k_vec);
                #pragma unroll
                for (int offset = VECS_PER_HEAD / 2; offset > 0; offset /= 2) score += __shfl_xor(score, offset, WARP_SIZE);
                score = __shfl(score, (tid / VECS_PER_HEAD) * VECS_PER_HEAD, WARP_SIZE); 
                
                float m_prev = m_i;
                m_i = fmaxf(m_i, score);
                float exp_score = expf(score - m_i);
                float correction = expf(m_prev - m_i);
                l_i = l_i * correction + exp_score;
                #pragma unroll
                for(int i=0; i<8; ++i) acc[i] = acc[i] * correction + static_cast<float>(v_vec.bf16[i]) * exp_score;
            }
        }
    }

    // B. REMAINDER LOOP
    if (end_blk == total_logical_blocks && start_blk < total_logical_blocks) {
        int blk_idx = total_logical_blocks - 1;
        const int phys_block = my_block_table[blk_idx];
        #pragma unroll (HEAD_SIZE >= 256 ? 4 : 16)
        for (int t = 0; t < BLOCK_SIZE; t += TOKENS_PER_WARP) {
            int current_token_off = t + token_idx_in_group;
            int global_kv_idx = blk_idx * BLOCK_SIZE + current_token_off;
            if (!((current_token_off < BLOCK_SIZE) && (global_kv_idx < kv_len) && (global_kv_idx <= causal_limit))) continue;

            long long k_offset = k_head_offset + (long long)phys_block * k_stride_block + (long long)current_token_off * k_stride_token;
            const float4* k_ptr = reinterpret_cast<const float4*>(k_cache + k_offset);
            Vec128 k_vec; k_vec.f4 = k_ptr[vec_idx];

            float score = vec_dot(q_vec, k_vec);
            #pragma unroll
            for (int offset = VECS_PER_HEAD / 2; offset > 0; offset /= 2) score += __shfl_xor(score, offset, WARP_SIZE);
            score = __shfl(score, (tid / VECS_PER_HEAD) * VECS_PER_HEAD, WARP_SIZE); 
            
            long long v_offset = v_head_offset + (long long)phys_block * v_stride_block + (long long)current_token_off * v_stride_token;
            const float4* v_ptr = reinterpret_cast<const float4*>(v_cache + v_offset);
            Vec128 v_vec; v_vec.f4 = v_ptr[vec_idx]; 
            
            float m_prev = m_i;
            m_i = fmaxf(m_i, score);
            float exp_score = expf(score - m_i);
            float correction = expf(m_prev - m_i);
            l_i = l_i * correction + exp_score;
            #pragma unroll
            for(int i=0; i<8; ++i) acc[i] = acc[i] * correction + static_cast<float>(v_vec.bf16[i]) * exp_score;
        }
    }

    // --- 5. Reduction ---
    float global_m = m_i;
    #pragma unroll
    for (int offset = VECS_PER_HEAD; offset < WARP_SIZE; offset *= 2) global_m = fmaxf(global_m, __shfl_xor(global_m, offset, WARP_SIZE));
    float correction_factor = expf(m_i - global_m);
    float l_corrected = l_i * correction_factor;
    float global_sum_l = l_corrected;
    #pragma unroll
    for (int offset = VECS_PER_HEAD; offset < WARP_SIZE; offset *= 2) global_sum_l += __shfl_xor(global_sum_l, offset, WARP_SIZE);
    #pragma unroll
    for(int i=0; i<8; ++i) acc[i] *= correction_factor;
    #pragma unroll
    for(int i=0; i<8; ++i) {
        #pragma unroll
        for (int offset = VECS_PER_HEAD; offset < WARP_SIZE; offset *= 2) acc[i] += __shfl_xor(acc[i], offset, WARP_SIZE);
    }
    
    // --- 6. Output ---
    if (token_idx_in_group == 0 && vec_idx < VECS_PER_HEAD) {
        if (num_splits == 1) {
            float inv_sum = 1.0f / (global_sum_l + 1e-6f);
            float4* out_ptr = reinterpret_cast<float4*>(output + q_token_global_idx * num_q_heads * HEAD_SIZE + head_idx * HEAD_SIZE);
            Vec128 out_vec;
            #pragma unroll
            for(int i=0; i<8; ++i) out_vec.bf16[i] = hip_bfloat16(acc[i] * inv_sum);
            out_ptr[vec_idx] = out_vec.f4;
        } else {
            if (vec_idx == 0) {
                 int meta_idx = (q_token_global_idx * num_q_heads * num_splits + head_idx * num_splits + split_idx) * 2;
                 temp_meta[meta_idx] = global_m;
                 temp_meta[meta_idx + 1] = global_sum_l;
            }
            int base_offset = (q_token_global_idx * num_q_heads * num_splits + head_idx * num_splits + split_idx) * HEAD_SIZE;
            Vec128 tmp_store;
            #pragma unroll
            for(int i=0; i<8; ++i) tmp_store.bf16[i] = hip_bfloat16(acc[i]); 
            float4* temp_ptr_f4 = reinterpret_cast<float4*>(temp_acc + base_offset);
            temp_ptr_f4[vec_idx] = tmp_store.f4;
        }
    }
}

// --- Merge Kernel ---
template<int HEAD_SIZE>
__global__ void split_merge_kernel(
    hip_bfloat16* __restrict__ output,
    const hip_bfloat16* __restrict__ temp_acc, 
    const float* __restrict__ temp_meta,
    int num_splits,
    int num_q_heads
) {
    const int q_idx = blockIdx.x;
    const int h_idx = blockIdx.y;
    const int tid = threadIdx.x;
    if (tid >= HEAD_SIZE) return;

    float g_max = -1e20f;
    int base_meta = (q_idx * num_q_heads + h_idx) * num_splits;
    for(int s=0; s<num_splits; ++s) {
        float m = temp_meta[(base_meta + s) * 2];
        if (m > -1e19f) g_max = fmaxf(g_max, m);
    }
    
    float g_sum = 0.0f;
    float final_val = 0.0f;
    int base_acc = (q_idx * num_q_heads + h_idx) * num_splits * HEAD_SIZE;

    for(int s=0; s<num_splits; ++s) {
        float m = temp_meta[(base_meta + s) * 2];
        float sum = temp_meta[(base_meta + s) * 2 + 1];
        if (m > -1e19f) {
            float weight = sum * expf(m - g_max);
            g_sum += weight;
            float val = static_cast<float>(temp_acc[base_acc + s * HEAD_SIZE + tid]);
            final_val += val * expf(m - g_max);
        }
    }
    
    int out_idx = q_idx * num_q_heads * HEAD_SIZE + h_idx * HEAD_SIZE + tid;
    output[out_idx] = hip_bfloat16(final_val / (g_sum + 1e-6f));
}

// --- Host Function ---
Tensor custom_paged_attn(
    const Tensor& query, const Tensor& key_cache, const Tensor& value_cache,
    const std::vector<int64_t>& query_lens, const std::vector<int64_t>& kv_lens,
    Tensor block_tables, double scale, double soft_cap = -1.0
) {
    const int total_q = query.size(0);
    const int num_q_heads = query.size(1);
    const int head_size = query.size(2);
    const int num_kv_heads = key_cache.size(2);
    const int max_blocks_per_seq = block_tables.size(1);
    
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();
    Tensor output = torch::empty(query.sizes(), query.options());

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(query.device());
    Tensor q_lens_gpu = torch::empty({(long)query_lens.size()}, opts_i64);
    Tensor kv_lens_gpu = torch::empty({(long)kv_lens.size()}, opts_i64);
    
    hipMemcpyAsync(q_lens_gpu.data_ptr(), query_lens.data(), query_lens.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(kv_lens_gpu.data_ptr(), kv_lens.data(), kv_lens.size() * sizeof(int64_t), hipMemcpyHostToDevice, stream);

    // ===============================================
    // V16 Strategy: Template Specialization Decision
    // ===============================================
    
    // Heuristic 1: Choose Number of Warps per Block
    // If we have few query tokens (BS < 8), we need max occupancy -> 1 Warp/Block.
    // If we have many query tokens (BS >= 8), we need latency hiding -> 4 Warps/Block.
    bool use_multi_warp = (total_q >= 8);
    int warps_per_block = use_multi_warp ? 4 : 1;

    // Heuristic 2: Choose Number of Splits
    int avg_kv_len = 0;
    if (kv_lens.size() > 0) avg_kv_len = kv_lens[0]; 
    
    int num_splits = 1;
    // Note: If using multi-warp, effective blocks = total_q * num_q_heads / 4
    int blocks_existing = (total_q * num_q_heads + warps_per_block - 1) / warps_per_block;
    
    // Target blocks depends on strategy
    // If 1-Warp: Target 2048+ to fill GPU with many small blocks
    // If 4-Warp: Target 512+ is usually enough (V9 showed this)
    int target_blocks = use_multi_warp ? 512 : 2048; 
    
    if (blocks_existing < target_blocks && avg_kv_len > 128) {
        num_splits = target_blocks / blocks_existing;
        int max_splits = (head_size >= 256) ? 16 : 8; 
        if (num_splits > max_splits) num_splits = max_splits;
        if (num_splits < 1) num_splits = 1;
    }

    Tensor temp_acc, temp_meta;
    if (num_splits > 1) {
        auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
        auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(query.device());
        temp_acc = torch::empty({total_q, num_q_heads, num_splits, head_size}, opts_bf16);
        temp_meta = torch::empty({total_q, num_q_heads, num_splits, 2}, opts_f32);
    } else {
        temp_acc = torch::empty({0}, query.options());
        temp_meta = torch::empty({0}, query.options());
    }

    int grid_y = (num_q_heads + warps_per_block - 1) / warps_per_block;
    dim3 grid(total_q, grid_y, num_splits);
    dim3 block(WARP_SIZE, warps_per_block);
    
    // MACRO to launch specific template instance
    #define LAUNCH_INSTANTIATED_KERNEL(H_SIZE, W_COUNT) \
        paged_attn_kernel_opt<H_SIZE, 16, W_COUNT><<<grid, block, 0, stream>>>( \
            (hip_bfloat16*)output.data_ptr(), \
            num_splits > 1 ? (hip_bfloat16*)temp_acc.data_ptr() : nullptr, \
            num_splits > 1 ? temp_meta.data_ptr<float>() : nullptr, \
            (hip_bfloat16*)query.data_ptr(), \
            (hip_bfloat16*)key_cache.data_ptr(), \
            (hip_bfloat16*)value_cache.data_ptr(), \
            block_tables.data_ptr<int>(), \
            kv_lens_gpu.data_ptr<int64_t>(), \
            q_lens_gpu.data_ptr<int64_t>(), \
            (float)scale, \
            max_blocks_per_seq, \
            query.stride(0), query.stride(1), \
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), \
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), \
            num_kv_heads, \
            num_splits, \
            num_q_heads \
        )

    #define DISPATCH_HEAD(H_SIZE) \
        if (use_multi_warp) { \
            LAUNCH_INSTANTIATED_KERNEL(H_SIZE, 4); \
        } else { \
            LAUNCH_INSTANTIATED_KERNEL(H_SIZE, 1); \
        }

    switch (head_size) {
        case 128: DISPATCH_HEAD(128); break;
        case 256: DISPATCH_HEAD(256); break;
    }
    
    if (num_splits > 1) {
        dim3 merge_grid(total_q, num_q_heads);
        dim3 merge_block(head_size);
        
        // Simple merge dispatch
    
        if (head_size == 128) split_merge_kernel<128><<<merge_grid, merge_block, 0, stream>>>((hip_bfloat16*)output.data_ptr(), (hip_bfloat16*)temp_acc.data_ptr(), temp_meta.data_ptr<float>(), num_splits, num_q_heads);
        else if (head_size == 256) split_merge_kernel<256><<<merge_grid, merge_block, 0, stream>>>((hip_bfloat16*)output.data_ptr(), (hip_bfloat16*)temp_acc.data_ptr(), temp_meta.data_ptr<float>(), num_splits, num_q_heads);
    }

    return output;
}

PYBIND11_MODULE(custom_attn, m) {
    m.def("custom_paged_attn", &custom_paged_attn, "Optimized Paged Attention V16 (Template Specialized)");
}