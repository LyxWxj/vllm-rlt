"""Single-query paged attention with independent block tables for every row.

Rows may represent different requests, positions, or recurrence depths. The
manager selects each row's depth-specific block table before dispatching here.
"""

import math

import torch


def torch_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference attention over [physical block, token, KV head, dimension]."""
    if (current_key is None) != (current_value is None):
        raise ValueError("current_key and current_value must be supplied together")
    output = torch.empty_like(q)
    block_size = key_cache.shape[1]
    head_groups = q.shape[1] // key_cache.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    for row, length in enumerate(context_lengths.tolist()):
        if length == 0:
            output[row].zero_()
            continue
        history_length = length - 1 if current_key is not None else length
        token_positions = torch.arange(history_length, device=q.device)
        blocks = block_tables[row, token_positions // block_size]
        offsets = token_positions % block_size
        keys = key_cache[blocks, offsets]
        values = value_cache[blocks, offsets]
        if current_key is not None:
            keys = torch.cat((keys, current_key[row].unsqueeze(0)))
            values = torch.cat((values, current_value[row].unsqueeze(0)))
        keys = keys.repeat_interleave(head_groups, dim=1)
        values = values.repeat_interleave(head_groups, dim=1)
        scores = torch.einsum("hd,thd->ht", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum("ht,thd->hd", probabilities, values.float()).to(q.dtype)
    return output


def triton_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch explicitly to Triton, with no silent reference fallback."""
    if q.device.type != "cuda":
        raise ValueError("the Triton attention backend requires a CUDA or ROCm device")
    if q.shape[-1] > 256:
        raise ValueError("the Triton attention backend supports head_dim <= 256")
    # Keep Triton optional and avoid loading its runtime for CPU-only callers.
    from vllm_rlt.kernels.triton_attention import paged_attention

    return paged_attention(
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        current_key=current_key,
        current_value=current_value,
    )
