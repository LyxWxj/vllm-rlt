"""Paged GQA kernel using bounded-memory online softmax.

One Triton program computes one query head. Unlike a fixed-depth batch, every
query row receives its own physical block table. This is a correctness-first
kernel without a performance claim or autotuning.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    CURRENT_K,
    CURRENT_V,
    OUT,
    q_batch_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    q_dim_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    k_dim_stride: tl.constexpr,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    v_dim_stride: tl.constexpr,
    table_stride: tl.constexpr,
    current_k_batch_stride: tl.constexpr,
    current_k_head_stride: tl.constexpr,
    current_k_dim_stride: tl.constexpr,
    current_v_batch_stride: tl.constexpr,
    current_v_head_stride: tl.constexpr,
    current_v_dim_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    out_head_stride: tl.constexpr,
    out_dim_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    USE_CURRENT: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // HEAD_GROUPS
    length = tl.load(LENGTHS + row)
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        Q + row * q_batch_stride + head * q_head_stride + dims * q_dim_stride,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(0, length, BLOCK_T):
        positions = start + tl.arange(0, BLOCK_T)
        valid_tokens = positions < length
        current_positions = valid_tokens & (positions == length - 1)
        cache_positions = valid_tokens & ~current_positions if USE_CURRENT else valid_tokens
        blocks = tl.load(
            TABLES + row * table_stride + positions // PAGE_SIZE,
            mask=cache_positions,
            other=0,
        ).to(tl.int64)
        # Physical IDs fit in int32, but ID * block_stride can exceed 2**31
        # elements in a large KV pool. Promote BEFORE multiplying, for K and V.
        offsets = positions % PAGE_SIZE
        keys = tl.load(
            K
            + blocks[:, None] * k_block_stride
            + offsets[:, None] * k_token_stride
            + kv_head * k_head_stride
            + dims[None, :] * k_dim_stride,
            mask=cache_positions[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        if USE_CURRENT:
            current_keys = tl.load(
                CURRENT_K
                + row * current_k_batch_stride
                + kv_head * current_k_head_stride
                + dims * current_k_dim_stride,
                mask=dims < HEAD_DIM,
                other=0,
            ).to(tl.float32)
            keys = tl.where(current_positions[:, None], current_keys[None, :], keys)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        correction = tl.exp(maximum - next_maximum)
        weights = tl.exp(scores - next_maximum)
        values = tl.load(
            V
            + blocks[:, None] * v_block_stride
            + offsets[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :] * v_dim_stride,
            mask=cache_positions[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        if USE_CURRENT:
            current_values = tl.load(
                CURRENT_V
                + row * current_v_batch_stride
                + kv_head * current_v_head_stride
                + dims * current_v_dim_stride,
                mask=dims < HEAD_DIM,
                other=0,
            ).to(tl.float32)
            values = tl.where(current_positions[:, None], current_values[None, :], values)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        normalizer = normalizer * correction + tl.sum(weights, axis=0)
        maximum = next_maximum
    tl.store(
        OUT + row * out_batch_stride + head * out_head_stride + dims * out_dim_stride,
        accumulator / tl.maximum(normalizer, 1.0e-30),
        mask=dims < HEAD_DIM,
    )


def paged_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    *,
    current_key=None,
    current_value=None,
):
    """Launch over [batch row, query head]; inputs are validated by the manager."""
    if (current_key is None) != (current_value is None):
        raise ValueError("current_key and current_value must be supplied together")
    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    if current_key is None:
        current_key = key_cache
        current_value = value_cache
        current_k_strides = (0, 0, 0)
        current_v_strides = (0, 0, 0)
    else:
        current_k_strides = current_key.stride()
        current_v_strides = current_value.stride()
    _paged_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        current_key,
        current_value,
        output,
        *q.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        block_tables.stride(0),
        *current_k_strides,
        *current_v_strides,
        *output.stride(),
        HEAD_DIM=q.shape[-1],
        HEAD_GROUPS=q.shape[1] // key_cache.shape[2],
        PAGE_SIZE=key_cache.shape[1],
        SCALE=q.shape[-1] ** -0.5,
        BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        BLOCK_T=32,
        USE_CURRENT=current_k_strides != (0, 0, 0),
        num_warps=4,
    )
    return output
