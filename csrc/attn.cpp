#include <torch/torch.h>
#include "hip/hip_runtime.h"
#include <vector>
#include <limits>

using torch::Tensor;
using namespace torch::indexing;

// TODO: Write your kernels here to optimized the custom_paged_attn op. 
// Note: You're not allowed to use PyTorch built-in function (e.g., einsum, softmax, ... except for torch.empty(...)), 
// meaning you'll have to replace them with your C++ implementation.

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
    TORCH_CHECK(query_lens.size() == kv_lens.size(),
                "query_lens and kv_lens must have same size");

    const int64_t num_seqs = static_cast<int64_t>(query_lens.size());

    Tensor block_tables_cpu =
        block_tables.to(torch::kCPU).to(torch::kLong);

    const int64_t block_size   = key_cache.size(1);
    const int64_t num_kv_heads = key_cache.size(2);
    const int64_t head_size    = key_cache.size(3);

    std::vector<Tensor> outputs;
    outputs.reserve(num_seqs);

    int64_t start_idx = 0;

    auto device = query.device();
    auto dtype  = query.dtype();

    for (int64_t i = 0; i < num_seqs; ++i) {
        const int64_t query_len = query_lens[i];
        const int64_t kv_len    = kv_lens[i];

        // q = query[start_idx : start_idx + query_len]; q *= scale
        Tensor q = query.narrow(/*dim=*/0,
                                /*start=*/start_idx,
                                /*length=*/query_len) * scale;

        // num_kv_blocks = (kv_len + block_size - 1) // block_size
        const int64_t num_kv_blocks = (kv_len + block_size - 1) / block_size;

        // block_indices = block_tables[i, :num_kv_blocks]
        using torch::indexing::Slice;
        Tensor block_indices_cpu =
            block_tables_cpu.index({i, Slice(0, num_kv_blocks)});
        // Move indices to same device as key_cache / value_cache
        Tensor block_indices =
            block_indices_cpu.to(key_cache.device(), /*non_blocking=*/false, /*copy=*/true);

        // k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        Tensor k_blocks = key_cache.index_select(/*dim=*/0, block_indices).contiguous();
        Tensor k = k_blocks.view({-1, num_kv_heads, head_size})
                           .narrow(/*dim=*/0, /*start=*/0, /*length=*/kv_len);

        // v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        Tensor v_blocks = value_cache.index_select(/*dim=*/0, block_indices).contiguous();
        Tensor v = v_blocks.view({-1, num_kv_heads, head_size})
                           .narrow(/*dim=*/0, /*start=*/0, /*length=*/kv_len);

        // If num_heads != num_kv_heads: repeat kv heads
        if (q.size(1) != k.size(1)) {
            const int64_t repeat = q.size(1) / k.size(1);
            // Equivalent to torch.repeat_interleave(..., dim=1)
            k = torch::repeat_interleave(k, repeat, /*dim=*/1);
            v = torch::repeat_interleave(v, repeat, /*dim=*/1);
        }

        // attn = einsum("qhd,khd->hqk", q, k).float()
        Tensor attn = torch::einsum("qhd,khd->hqk", {q, k}).to(torch::kFloat);

        // empty_mask = ones(query_len, kv_len); mask = triu(..., diag=kv_len-query_len+1).bool()
        auto mask_opts = torch::TensorOptions().device(device).dtype(torch::kFloat);
        Tensor empty_mask = torch::ones({query_len, kv_len}, mask_opts);
        Tensor mask = torch::triu(empty_mask, kv_len - query_len + 1).to(torch::kBool);

        // Soft cap if requested
        if (soft_cap > 0.0) {
            attn = soft_cap * torch::tanh(attn / soft_cap);
        }

        // attn.masked_fill_(mask, -inf)
        attn.masked_fill_(mask, -std::numeric_limits<float>::infinity());

        // attn = softmax(attn, dim=-1).to(v.dtype)
        attn = torch::softmax(attn, /*dim=*/-1).to(v.dtype());

        // out = einsum("hqk,khd->qhd", attn, v)
        Tensor out = torch::einsum("hqk,khd->qhd", {attn, v});

        outputs.push_back(out);
        start_idx += query_len;
    }

    Tensor result = torch::cat(outputs, /*dim=*/0);
    return result;
}

PYBIND11_MODULE(custom_attn, m) {
    m.def("custom_paged_attn", &custom_paged_attn,
          "Launch the custom paged attn with PyTorch tensors.");
}
