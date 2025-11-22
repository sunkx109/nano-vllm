import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride, # num_heads * head_dim
    value_ptr,
    value_stride,# num_heads * head_dim
    k_cache_ptr, #[num_kvcache_blocks,block_size , D]
    v_cache_ptr, #[num_kvcache_blocks,block_size , D]
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0) # seq_token_idx
    # 前面我们说过slot_mapping存的是每个token在整个庞大的block cache里面的绝对位置索引
    # 取出每个token所对应的实际位置slot
    # 这个绝对位置的范围是[0,num_kvcache_blocks*block_size)
    # 也就是每层 k_cache_ptr 和 v_cache_ptr的总大小
    slot = tl.load(slot_mapping_ptr + idx) 
    if slot == -1: return # 如果取出来是-1 就表示这个token不做处理
    # 计算待存入的KV的offset，待存入的KV是前层计算的结果是连续的
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 取KV
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 通过slot_mapping计算当前KV的存储位置
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1 # 确保Key Value的最后一个维度是连续的
    assert key.stride(1) == head_dim and value.stride(1) == head_dim # 第二维的跨度等于head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D # k_cache和v_cache的第二维跨度等于D
    assert slot_mapping.numel() == N # slot_mapping记录的是所有seq内部token在整个block cache中的绝对位置
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # 在model_runner中会调用allocate_kv_cache来初始分配k_cache和v_cache

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            # 因为在modelrun构造时就创建了KV，所以这个分支始终会走到
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                # 如果context.block_tables不为空, 那说明有前缀缓存
                # 这时K V应该是完整的KV cache，而非输入的KV(因为输入的KV只是新增的部分)
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
