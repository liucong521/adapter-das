# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding kernel mirroring sgl_kernel.rotary_embedding (pos_enc.cu).

    rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

In-place on query/key, rows indexed by token:
    for token i at position p = positions[i], every head vector is rotated by
    cos/sin of p. query/key are (num_tokens, num_heads * head_size), cache is
    (max_pos, rotary_dim) fp32 as built by sglang's RotaryEmbedding layer.

Layouts (both must match the CUDA kernel exactly):
    is_neox=True  — NeoX/rotate_half: pair x[i] with x[i + rotary_dim/2];
                    cache row = [cos(rotary_dim/2) | sin(rotary_dim/2)].
    is_neox=False — GPT-J interleaved: pair even lane 2j with odd lane 2j+1;
                    cache row = [cos0, sin0, cos1, sin1, ...] interleaved.
"""

from __future__ import annotations

import torch

from .compat import HAS_TRITON, check_cuda_tensor

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_neox_kernel(
        POS,  # (num_tokens,) int64
        T,  # (num_tokens, num_heads * head_size) row-major, updated in place
        t_stride,
        COS_SIN,  # (max_pos, rotary_dim) fp32
        num_heads,
        HEAD_SIZE: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        head_blk = tl.program_id(1)
        half: tl.constexpr = ROTARY_DIM // 2
        half_idx = tl.arange(0, half)

        pos = tl.load(POS + token)
        c = tl.load(COS_SIN + pos * ROTARY_DIM + half_idx)
        s = tl.load(COS_SIN + pos * ROTARY_DIM + half + half_idx)

        heads = head_blk * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
        hmask = heads < num_heads
        off = tl.where(hmask, heads, 0) * HEAD_SIZE

        base = T + token * t_stride
        x1 = tl.load(base + off[:, None] + half_idx[None, :],
                     mask=hmask[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(base + off[:, None] + (half + half_idx)[None, :],
                     mask=hmask[:, None], other=0.0).to(tl.float32)
        y1 = x1 * c[None, :] - x2 * s[None, :]
        y2 = x1 * s[None, :] + x2 * c[None, :]
        tl.store(base + off[:, None] + half_idx[None, :],
                 y1.to(base.dtype.element_ty), mask=hmask[:, None])
        tl.store(base + off[:, None] + (half + half_idx)[None, :],
                 y2.to(base.dtype.element_ty), mask=hmask[:, None])


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Rotate query (and key) rows in place. See module docstring."""
    if positions.numel() == 0:
        return
    use_triton = HAS_TRITON and query.is_cuda and cos_sin_cache.is_cuda and is_neox
    if key is not None and use_triton:
        use_triton = key.is_cuda
    if not use_triton:
        impl = (
            _rotary_embedding_neox_torch
            if is_neox
            else _rotary_embedding_interleaved_torch
        )
        impl(positions, query, key, head_size, cos_sin_cache)
        return

    check_cuda_tensor(positions, query, cos_sin_cache)
    if key is not None:
        check_cuda_tensor(key)

    rotary_dim = cos_sin_cache.shape[1]
    num_tokens = positions.shape[0]
    contiguous_in = query.is_contiguous() and (key is None or key.is_contiguous())
    q = query if query.is_contiguous() else query.contiguous()
    k = None
    if key is not None:
        k = key if key.is_contiguous() else key.contiguous()

    BLOCK_HEADS = 8
    q_heads = q.shape[-1] // head_size
    grid = (num_tokens, triton.cdiv(q_heads, BLOCK_HEADS))
    _rope_neox_kernel[grid](
        positions,
        q,
        q.stride(0),
        cos_sin_cache,
        q_heads,
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        BLOCK_HEADS=BLOCK_HEADS,
        num_warps=4,
    )
    if k is not None and k.data_ptr() != q.data_ptr():
        k_heads = k.shape[-1] // head_size
        grid_k = (num_tokens, triton.cdiv(k_heads, BLOCK_HEADS))
        _rope_neox_kernel[grid_k](
            positions,
            k,
            k.stride(0),
            cos_sin_cache,
            k_heads,
            HEAD_SIZE=head_size,
            ROTARY_DIM=rotary_dim,
            BLOCK_HEADS=BLOCK_HEADS,
            num_warps=4,
        )

    if not contiguous_in:
        if q.data_ptr() != query.data_ptr():
            query.copy_(q.view(query.shape))
        if k is not None and k.data_ptr() != key.data_ptr():
            key.copy_(k.view(key.shape))


def _rotary_embedding_neox_torch(positions, query, key, head_size, cos_sin_cache):
    """Pure-torch NeoX reference (also the CPU path)."""
    rotary_dim = cos_sin_cache.shape[1]
    pos = positions.long()
    cos = cos_sin_cache[pos, : rotary_dim // 2].float()
    sin = cos_sin_cache[pos, rotary_dim // 2:].float()
    num_tokens = positions.shape[0]

    def _rope(t):
        if t is None:
            return None
        orig_shape = t.shape
        t = t.view(num_tokens, -1, head_size)
        x1 = t[..., : rotary_dim // 2].float()
        x2 = t[..., rotary_dim // 2 : rotary_dim].float()
        y1 = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
        y2 = x2 * cos.unsqueeze(1) + x1 * sin.unsqueeze(1)
        out = t.clone()
        out[..., : rotary_dim // 2] = y1.to(t.dtype)
        out[..., rotary_dim // 2 : rotary_dim] = y2.to(t.dtype)
        return out.view(orig_shape)

    out_q = _rope(query)
    if out_q is not None:
        query.copy_(out_q)
    out_k = _rope(key)
    if out_k is not None:
        key.copy_(out_k)


def _rotary_embedding_interleaved_torch(positions, query, key, head_size, cos_sin_cache):
    """GPT-J interleaved layout torch reference.

    sglang's RotaryEmbedding always stores the cache half-split
    ([cos(rotary_dim/2) | sin(rotary_dim/2)]), independent of
    is_neox_style; only the lane pairing of the rotated vector differs.
    Pair even lane 2j with odd lane 2j+1.
    """
    rotary_dim = cos_sin_cache.shape[1]
    pos = positions.long()
    cache = cos_sin_cache[pos].float()  # (T, rotary_dim)
    cos_i = cache[:, : rotary_dim // 2]
    sin_i = cache[:, rotary_dim // 2:]
    num_tokens = positions.shape[0]

    def _rope(t):
        if t is None:
            return None
        orig_shape = t.shape
        t = t.view(num_tokens, -1, head_size)
        x1 = t[..., :rotary_dim:2].float()  # even lanes
        x2 = t[..., 1:rotary_dim:2].float()  # odd lanes
        y1 = x1 * cos_i.unsqueeze(1) - x2 * sin_i.unsqueeze(1)
        y2 = x1 * sin_i.unsqueeze(1) + x2 * cos_i.unsqueeze(1)
        out = t.clone()
        out[..., :rotary_dim:2] = y1.to(t.dtype)
        out[..., 1:rotary_dim:2] = y2.to(t.dtype)
        return out.view(orig_shape)

    out_q = _rope(query)
    if out_q is not None:
        query.copy_(out_q)
    out_k = _rope(key)
    if out_k is not None:
        key.copy_(out_k)
