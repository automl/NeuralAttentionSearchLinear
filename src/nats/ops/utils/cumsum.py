# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.cumsum import chunk_local_cumsum_scalar
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import check_shared_mem, input_guard

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    #'N_NAtS_BLOCK_PER_T': lambda args: triton.cdiv(args['BT'], args['NAtS_BLOCK_SIZE']),
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT'])
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE']
)
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def chunk_local_cumsum_nats_scalar_kernel(
    s,
    o,
    scale,
    nats_block_types,
    nats_block_indices,
    cu_seqlens,
    cu_seqlens_nats,
    chunk_indices,
    chunk_indices_op_nats,
    T,
    TNAtS,
    B: tl.constexpr,
    H: tl.constexpr,
    HNAtS:tl.constexpr,
    GNAtS: tl.constexpr,
    BT: tl.constexpr,
    NAtS_BLOCK_SIZE: tl.constexpr,
    N_TYPES: tl.constexpr,
    OFFSET_OP: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
):
    #i_t, i_bh = tl.program_id(0), tl.program_id(1)
    #i_b, i_h = i_bh // H, i_bh % H

    i_t_, i_gnats = tl.program_id(0), tl.program_id(1)
    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t_ // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t_ % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t_
        i_t_nats_offset = 0
    off_bh_nats = tl.load(chunk_indices_op_nats + i_t_nats * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_op_nats + i_t_nats * 2 + 1).to(tl.int32)
    i_b = off_bh_nats // HNAtS
    i_hnats = off_bh_nats % HNAtS

    i_h = i_hnats * GNAtS + i_gnats

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    stride_block_types_t = N_TYPES * HNAtS
    o += i_t_ * BT * GNAtS + i_gnats

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t)
    o_t0 = b_o_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (o_t0,), (BT,), (0,))
        p_o = tl.make_block_ptr(o, (T-o_t0,), (1,), (0,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (o_t0,), (BT,), (0,))
        p_o = tl.make_block_ptr(o, (T-o_t0,), (H,), (0,), (BT,), (0,))

    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE']
)
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def chunk_local_cumsum_nats_vector_kernel(
    s,
    o,
    scale,
    nats_block_types,
    nats_block_indices,
    cu_seqlens,
    cu_seqlens_nats,
    chunk_indices,
    chunk_indices_op_nats,
    T,
    TNAtS,
    B: tl.constexpr,
    H: tl.constexpr,
    HNAtS: tl.constexpr,
    GNAtS: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    NAtS_BLOCK_SIZE: tl.constexpr,
    N_TYPES: tl.constexpr,
    OFFSET_OP: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    N_NAtS_BLOCK_PER_T: tl.constexpr,
):
    # TODO fix this for NATS_BLOCK_SIZE > BT!!!
    #i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    #i_b, i_h = i_bh // H, i_bh % H


    i_s, i_t_, i_hnats = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    off_bh_nats = tl.load(chunk_indices_op_nats + i_t_ * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_op_nats + i_t_ * 2 + 1).to(tl.int32)
    i_b = off_bh_nats // HNAtS
    i_hnats = off_bh_nats % HNAtS

    i_h = i_hnats * GNAtS + i_hnats

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    if HEAD_FIRST:
        stride_sh = T * S
        stride_st = S
    else:
        stride_sh = S
        stride_st = H * S
    stride_chunk_types_t = N_TYPES * HNAtS
    stride_ot = GNAtS

    s += bos * H * S + i_h * stride_sh + i_s * BS

    o += (i_t_ * BT * GNAtS + i_hnats) * S + i_s * BS

    #if HEAD_FIRST:
    #    p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    #    p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    #else:
    #    p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    #    p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    #b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE + tl.arange(0, N_NAtS_BLOCK_PER_T)
    nats_chunk_msk = load_idx_chunk < TNAtS
    b_delta_chunk_indices = tl.load(nats_block_indices + load_idx_chunk * stride_chunk_types_t,
                                   mask=nats_chunk_msk, other=TNAtS
                                   )
    b_delta_chunk_indices_scaled = b_delta_chunk_indices * NAtS_BLOCK_SIZE
    o_t0 = tl.reshape(
        b_delta_chunk_indices_scaled[:, None] + tl.arange(0, NAtS_BLOCK_SIZE)[None, :], [BT]
    )
    m_t = o_t0 < T
    o_s = tl.arange(0, BS)
    msk = m_t[:, None] & ((i_s * BS + o_s) < S)[None, :]
    b_s = tl.load(s + o_t0[:, None] * stride_st + o_s[None, :], mask=msk, other=0).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    if HAS_SCALE:
        b_o *= scale
    #tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(o + tl.arange(0, BT)[:, None] * stride_ot + o_s[None, :], b_o.to(o.dtype.element_ty), mask=msk)


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'IS_VARLEN', 'REVERSE']
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [16, 32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'S', 'IS_VARLEN', 'REVERSE']
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        if HAS_SCALE:
            b_c *= scale
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_z += tl.sum(b_s, 0)


def chunk_local_nats_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    nats_block_types: torch.Tensor,
    nats_block_indices: torch.Tensor,
    n_nats_blocks: torch.Tensor,
    chunk_indices_op_nats: torch.Tensor,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_nats: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    nats_block_size: int = 8,
    offset_op: int = 0,
    compute_incomplete_chunk_scores:bool = False
) -> torch.Tensor:
    if compute_incomplete_chunk_scores:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,output_dtype=output_dtype
        )
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    #NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)

    chunk_local_cumsum_nats_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_indices=chunk_indices,
        chunk_indices_op_nats=chunk_indices_op_nats,
        T=T,
        TNAtS=TNAtS,
        B=B,
        H=H,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        BT=BT,
        NAtS_BLOCK_SIZE=nats_block_size,
        N_TYPES=n_opts,
        HEAD_FIRST=head_first,
        OFFSET_OP=offset_op,
        REVERSE=reverse,
        COMPUTE_ALL_CHUNKS=compute_incomplete_chunk_scores,
    )
    return g


def chunk_local_nats_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    nats_block_types: torch.Tensor,
    nats_block_indices: torch.Tensor,
    n_nats_blocks: torch.Tensor,
    chunk_indices_op_nats: torch.Tensor,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_nats: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    nats_block_size: int = 8,
    offset_op: int = 0,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    #NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    _, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS
    grid = (len(chunk_indices_op_nats), GNAtS)

    #g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    g_org, g = g, torch.empty(len(chunk_indices_op_nats) * BT, GNAtS, S, device=g.device, dtype=output_dtype or g.dtype)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), len(chunk_indices_op_nats), GNAt)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_local_cumsum_nats_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_indices=chunk_indices,
        chunk_indices_op_nats=chunk_indices_op_nats,
        T=T,
        TNAtS=TNAtS,
        B=B,
        H=H,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        S=S,
        BT=BT,
        NAtS_BLOCK_SIZE=nats_block_size,
        N_TYPES=n_opts,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return g


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (N * H,)
    chunk_global_cumsum_scalar_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return z


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, triton.next_power_of_2(S))

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (triton.cdiv(S, BS), N * H)
    chunk_global_cumsum_vector_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        S=S,
        BS=BS,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return z


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype
        )
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype
        )
    else:
        raise ValueError(
            f"Unsupported input shape {s.shape}, "
            f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
            f"or [B, H, T]/[B, H, T, D] otherwise"
        )


@input_guard
def chunk_local_nats_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    nats_block_types: torch.Tensor,
    nats_block_indices: torch.Tensor,
    n_nats_blocks: torch.Tensor,
    chunk_indices_op_nats: torch.Tensor,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    nats_block_size: int = 8,
    offset_op: int = 0,
    compute_incomplete_chunk_scores: bool=False,
    **kwargs
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_nats_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            chunk_indices_op_nats=chunk_indices_op_nats,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            nats_block_size=nats_block_size,
            offset_op=offset_op,
            compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        )
    elif len(g.shape) == 4:
        return chunk_local_nats_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            chunk_indices_op_nats=chunk_indices_op_nats,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            nats_block_size=nats_block_size,
            offset_op=offset_op,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
