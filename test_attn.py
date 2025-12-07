from typing import Optional

import torch

from custom_attn import custom_paged_attn
from aiter.test_common import checkAllclose, run_perftest



######################################################
# Scheduler Simulator
######################################################
def scheduler_simulator(ISL, OSL, concurrency, scale, total_KV_budget, max_num_batched_tokens, custom_scheduler=False):

    full_decode_output = []
    prefill_output = []
    request_waiting_from_client = concurrency * (scale - 1)
    waiting = [ISL] * concurrency
    running = [] # (num_prefilled_token, num_decoded_tokens)
    total_KV_budget_remaining = total_KV_budget

    scheduler_step = 0
    tailing_count = 0
    while waiting or running or request_waiting_from_client > 0:
        # at the first step of online benchmark, only 1 request in waiting queue, so only schedule ISL tokens
        token_budget = max_num_batched_tokens if scheduler_step > 0 else ISL
        scheduler_output = []
        scheduler_step += 1
        is_full_decode = True
        prefill_state = []
        
        # print(f"Step {scheduler_step}: waiting={len(waiting)}, running={len(running)}, request_waiting_from_client={request_waiting_from_client}, total_KV_budget_remaining={total_KV_budget_remaining}, token_budget={token_budget}")

        # Custom scheduler logic: if we have a lot of KV budget remaining, try to schedule more waiting requests first        
        if custom_scheduler and (total_KV_budget_remaining - token_budget)/total_KV_budget > 0.2 and len(waiting) > 0:
            req_index = 0
            while req_index < len(waiting) and token_budget > 0:
                w = waiting[req_index]
                num_new_tokens = min(token_budget, w)

                if total_KV_budget_remaining < num_new_tokens:
                    break

                running.append((num_new_tokens, 0))
                scheduler_output.append(num_new_tokens)

                is_full_decode = False
                req_index += 1
                token_budget -= num_new_tokens
                total_KV_budget_remaining -= num_new_tokens

            waiting = waiting[req_index:]

        # First, try to fill up the running requests
        has_preempted = False
        req_index = 0
        while req_index < len(running) and token_budget > 0:
            r = running[req_index]
            if r[0] == ISL:
                token_needed = 1
            else:
                token_needed = ISL - r[0]
                assert r[1] == 0

            num_new_tokens = min(token_budget, token_needed)

            while True:
                    # simulate allocation of KV memory
                if total_KV_budget_remaining < num_new_tokens:
                    prempempted = running.pop()
                    total_KV_budget_remaining += prempempted[0] + prempempted[1]
                    waiting.append(ISL)
                    has_preempted = True
                        
                    if len(running) == req_index:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
                
            if not can_schedule:
                break

            if r[0] < ISL:
                assert r[0] + num_new_tokens <= ISL
                if r[0] + num_new_tokens == ISL:
                    running[req_index] = (ISL, 1)
                else:
                    running[req_index] = (r[0] + num_new_tokens, r[1])
                is_full_decode = False
                prefill_state.append((num_new_tokens, r[0]))
            else:
                running[req_index] = (r[0], r[1] + 1)

            token_budget -= num_new_tokens
            total_KV_budget_remaining -= num_new_tokens

            scheduler_output.append(num_new_tokens)

            req_index += 1

        # Next, try to fill up the waiting requests
        if not has_preempted:
            req_index = 0
            while req_index < len(waiting) and token_budget > 0:
                w = waiting[req_index]
                num_new_tokens = min(token_budget, w)

                if total_KV_budget_remaining < num_new_tokens:
                    break

                running.append((num_new_tokens, 0))
                scheduler_output.append(num_new_tokens)

                is_full_decode = False
                prefill_state.append((num_new_tokens, 0))
                req_index += 1
                token_budget -= num_new_tokens
                total_KV_budget_remaining -= num_new_tokens

            waiting = waiting[req_index:]    

        if is_full_decode:
            requests_state = [(1, r[0] + r[1] - 1) for r in running]
            full_decode_output.append(requests_state)
        else:
            if prefill_state:
                prefill_output.append(prefill_state)

        max_scheduled_token = max(scheduler_output) if scheduler_output else 0
        if max_scheduled_token == 1 and len(scheduler_output) < concurrency:
            tailing_count += 1

        # Handle Finished request
        previous = len(running)
        running = [(r[0], r[1]) for r in running if r[0] + r[1] < ISL + OSL]
        finished = previous - len(running)
        total_KV_budget_remaining += finished * (ISL + OSL)

        for _ in range(finished):
            if request_waiting_from_client > 0:
                waiting.append(ISL)
                request_waiting_from_client -= 1

    return full_decode_output, prefill_output

def get_scheduler_output(context_len, batch_size, OSL=512):
    ISL = context_len
    concurrency = batch_size
    scale = 1
    total_KV_budget = 1_836_352 * 8
    max_num_batched_tokens = 16384
    custom_scheduler = False

    return scheduler_simulator(ISL, OSL, concurrency, scale, total_KV_budget, max_num_batched_tokens, custom_scheduler)

######################################################
#  Prepare Inputs
######################################################
def create_inputs(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
    common_prefix_len: int = 0,
):
    query_lens = [l[0] for l in seq_lens]
    kv_lens = [l[1] for l in seq_lens]
    common_prefix_len = min(common_prefix_len, min(kv_lens)-1)
    num_seqs = len(query_lens)
    max_seqlen_q = max(query_lens)
    max_kv_len = max(kv_lens)
    max_seqlen_k = max_kv_len
    scale = head_size ** -0.5
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    cu_seqlens_q = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    if common_prefix_len > 0:
        common_prefix_len = (common_prefix_len // block_size) * block_size
        num_common_kv_blocks = common_prefix_len // block_size
    else:
        num_common_kv_blocks = 0

    number_of_unique_blocks = num_seqs * (max_num_blocks_per_seq - num_common_kv_blocks) + num_common_kv_blocks
    num_blocks = max(num_blocks, number_of_unique_blocks + 5)

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    kv_cache = torch.randn(num_blocks, 2, block_size, num_kv_heads, head_size, dtype=dtype)
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    # first block is used for masking for certain cases (load and ignore rather than masked load)
    # for better testing, but nan there to make sure those values do not propagate during attn. calc.
    # if it propagates to the results, tests will fail for sure when done this way
    key_cache[0] = float("nan")
    value_cache[0] = float("nan")

    # pool_value = torch.randperm(num_blocks-1)[:number_of_unique_blocks] + 1
    block_tables = torch.randint(
        1,
        num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    return (
        query,
        key_cache,
        value_cache,
        query_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_lens,
        max_seqlen_q,
        max_seqlen_k,
        block_tables,
        scale,
    )

def naive_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0

    FLOPS = 0
    BYTES = 0

    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        BYTES += q.numel() * q.element_size() # Q
        BYTES += k.numel() * k.element_size() # K
        BYTES += v.numel() * v.element_size() # V

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        FLOPS += 2 * q.shape[1] * q.shape[0] * k.shape[0] * q.shape[2]
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        FLOPS += 2 * attn.shape[0] * attn.shape[1] * attn.shape[2] * v.shape[2]

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0), FLOPS, BYTES


def run_vllm_unified(
    query,
    key_cache,
    value_cache,
    query_lens,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_lens,
    max_seqlen_q,
    max_seqlen_k,
    block_tables,
    scale,
):
    from vllm.attention.ops.triton_unified_attention import unified_attention as vllm_unified_attention
    output = torch.empty_like(query)
    vllm_unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=kv_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return output.squeeze()


########################################################
#  Benchmarking
########################################################
def bench_one_case(dtype, num_qheads: int, num_kvheads: int, head_size: int, seq_lens: list[tuple[int, int]], common_prefix_len: int = 0):

    block_size = 16
    num_blocks = 8192   # arbitrary, might be overide in create_inputs

    (
        query,
        key_cache,
        value_cache,
        query_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_lens,
        max_seqlen_q,
        max_seqlen_k,
        block_tables,
        scale,
    ) = create_inputs(
        seq_lens=seq_lens,
        num_heads=(num_qheads, num_kvheads),
        head_size=head_size,
        dtype=dtype,
        block_size=block_size,
        num_blocks=num_blocks,
        common_prefix_len=common_prefix_len,
    )

    # Note: naive_paged_attn should be called last because it modified "query"
    # clone query to avoid that

    custom_attn_output, custom_us = run_perftest(
        custom_paged_attn,
        query.clone(),
        key_cache,
        value_cache,
        query_lens,
        kv_lens.tolist(),
        block_tables,
        scale,
        0,
        num_iters=5,
        num_warmup=2,
    )

    (naive_attn_output, FLOPS, BYTES), naive_us = run_perftest(
        naive_paged_attn,
        query=query.clone(),
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        soft_cap=0,
        num_iters=5,
        num_warmup=2,
    )

    naive_tflops = FLOPS / naive_us / 1e6
    naive_bw = BYTES / naive_us / 1e6

    custom_tflops = FLOPS / custom_us / 1e6
    custom_bw = BYTES / custom_us / 1e6


    msg = f"\nNaive Attention: {naive_us:.2f} us, {naive_tflops:.2f} tflops, {naive_bw:.2f} TB/s\nCustom Attention: {custom_us:.2f} us, {custom_tflops:.2f} tflops, {custom_bw:.2f} TB/s"
    checkAllclose(custom_attn_output, naive_attn_output, atol=1e-2, rtol=1e-2, msg=msg)


def main():
    batch_size = [1, 8, 16, 32]
    context_len = [256, 2048]
    bench_prefill = False

    for dtype in [torch.bfloat16]:
        for num_qheads, num_kvheads, head_size in [
            (64, 1, 256),   # Step3 DP
            (64, 8, 128),   # GPT-OSS-120B DP
        ]:
            for b in batch_size[:]:
                for c in context_len[:]:
                    print(f"==== Batch Size {b}, Context Len {c} ====", flush=True)
                    scheduler_decode_outputs, scheduler_prefill_outputs = get_scheduler_output(context_len=c, batch_size=b)

                    if bench_prefill:
                        scheduler_outputs = scheduler_prefill_outputs
                    else:
                        scheduler_outputs = scheduler_decode_outputs

                    for i, seq_lens in enumerate(scheduler_outputs[:20]):
                        print(f"==== Case {i} ====")
                        bench_one_case(dtype, num_qheads, num_kvheads, head_size,
                                       seq_lens=seq_lens, common_prefix_len=0)
                        break


if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    main()
