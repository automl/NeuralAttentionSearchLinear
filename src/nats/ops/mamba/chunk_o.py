# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import pdb
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.common.chunk_o import chunk_bwd_dv_local
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT'])
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def chunk_fwd_kernel_o(
        q,
        k,
        v,
        h,
        g,
        #dt,
        o,
        nats_block_types,
        nats_block_indices,
        n_nats_blocks,
        cu_seqlens,
        cu_seqlens_nats,
        chunk_indices,
        starting_h_idx,
        nats_block_op_offsets,
        scale,
        T,
        TNAtS,
        H: tl.constexpr,
        HNAtS: tl.constexpr,
        GNAtS: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        NAtS_BLOCK_SIZE: tl.constexpr,
        N_TYPES: tl.constexpr,
        OFFSET_OP: tl.constexpr,
        N_TOKENS_IN_INIT_CHUNK: tl.constexpr,
        COMPUTE_INCOMPLETE_BLOCK_SCORES: tl.constexpr,
        DECAY_FOR_NON_MAMBA_BLOCKS:tl.constexpr,
        USE_G: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
        INCOMPLETE_BLOCK_WITH_START_HT: tl.constexpr,
):
    # This function is implemented for the cases where NAtS blocks do not algin with the BT chunks
    # Hence, sometimes we might need to do the computation twice since the BT chunks might involve
    # the values across 2 different NAtS Chunks

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t
        i_t_nats_offset = 0

    i_hnats = i_h // GNAtS
    i_gnats = i_h % GNAtS

    start_nats_kvh_t = tl.load(nats_block_op_offsets + (i_b * HNAtS + i_hnats)).to(tl.int32)
    end_nats_kvh_t = tl.load(nats_block_op_offsets + (i_b * HNAtS + i_hnats + 1)).to(tl.int32)

    # start_nats_h_t = tl.load(nats_block_op_offsets + 1 + 2 * (i_b * HNAtS + i_hnats)).to(tl.int32)
    # end_nats_h_t = tl.load(nats_block_op_offsets + 1 + 2 * (i_b * HNAtS + i_hnats + 1)).to(tl.int32)
    T_nats_blocks_compute = end_nats_kvh_t - start_nats_kvh_t

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_b).to(tl.int32), tl.load(cu_seqlens_nats + i_b + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
        starting_h_idx += i_b + i_b * tl.cdiv(T, BT) * HNAtS + i_hnats
        i_th = tl.load(starting_h_idx + i_t * HNAtS).to(tl.int32)
        i_th_last = tl.load(starting_h_idx + (TNAtS - 1) * HNAtS).to(tl.int32)

    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

        starting_h_idx += i_b * tl.cdiv(T, BT) * HNAtS + i_hnats
        i_th = tl.load(starting_h_idx + i_t * HNAtS).to(tl.int32)
        i_th_last = tl.load(starting_h_idx + (TNAtS - 1) * HNAtS).to(tl.int32)

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    # v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    # h += (i_tg * H + i_h).to(tl.int64) * K*V

    h += ((start_nats_kvh_t.to(tl.int64) + i_th) * GNAtS + i_gnats) * K * V

    stride_kt = K * H
    stride_nats_block = N_TYPES * HNAtS

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE

    # the condition for loading ht instead of h:
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    if COMPUTE_INCOMPLETE_BLOCK_SCORES:
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        v += (bos * H + i_h) * V
        if INCOMPLETE_BLOCK_WITH_START_HT:
            chunk_is_mamba = tl.load(nats_block_types + load_idx_chunk * N_TYPES * HNAtS).to(tl.int1)
            b_v_updated = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT - N_TOKENS_IN_INIT_CHUNK, i_k * BK), (BT, BK), (1, 0))

        # p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        # b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if COMPUTE_INCOMPLETE_BLOCK_SCORES:
            # we only need kv if we compute incomplete blocks with delta net
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_A += tl.dot(b_q, b_k)
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]

    if USE_G:
        #if WV_ARE_FLATTENED:
            #nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP
            #i_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_nats_block)
            #i_t0 = i_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT

        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        #dt += bos * H + i_h
        #p_dt = tl.make_block_ptr(dt, (T,), (H,), (i_t * BT,), (BT,), (0,))
        #b_dt = tl.load(p_dt, boundary_check=(0,))
        b_o = b_o * tl.exp(b_g)[:, None]
        if COMPUTE_INCOMPLETE_BLOCK_SCORES:
            b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])
            #b_A = b_A * b_dt

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    if COMPUTE_INCOMPLETE_BLOCK_SCORES:
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    else:
        b_o = b_o * scale

    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def chunk_mamba_fwd_nats_o(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: Optional[torch.Tensor],
        #dt: Optional[torch.Tensor],
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        chunk_indices_op_nats,
        nats_block_op_offsets: torch.Tensor,
        starting_h_idx: torch.Tensor,
        scale: Optional[float] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        nats_block_size: int = 8,
        offset_op: int = 0,
        n_tokens_in_init_chunk:int=0,
        compute_incomplete_block_scores: bool = False,
        incomplete_block_start_with_ht: bool = True,
        decay_for_non_mamba_blocks:bool=False,
) -> torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    _, TNAtS, HNAtS, n_opts = nats_block_indices.shape

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # o = torch.empty_like(v)
    o = torch.empty([B, T, H, V], dtype=v.dtype, device=v.device)

    # ht is also required here as the last state corresponds to the last "valid" hiden states, not the real
    # last state. For instance, if the we have 8 h values but the last valid delta chunk is the 5th chunk, then
    # the 6,7,8 chunks need to use the information from the hidden states (which should corresponds to the 6th
    # hidden state, but we already run out of the budgets for that)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        #dt=dt,
        o=o,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_indices=chunk_indices,
        starting_h_idx=starting_h_idx,
        nats_block_op_offsets=nats_block_op_offsets,
        scale=scale,
        T=T,
        TNAtS=TNAtS,
        H=H,
        HNAtS=HNAtS,
        GNAtS=H // HNAtS,
        K=K,
        V=V,
        BT=BT,
        NAtS_BLOCK_SIZE=nats_block_size,
        N_TYPES=n_opts,
        OFFSET_OP=offset_op,
        N_TOKENS_IN_INIT_CHUNK=n_tokens_in_init_chunk,
        COMPUTE_INCOMPLETE_BLOCK_SCORES=compute_incomplete_block_scores,
        DECAY_FOR_NON_MAMBA_BLOCKS=decay_for_non_mamba_blocks,
        INCOMPLETE_BLOCK_WITH_START_HT=incomplete_block_start_with_ht,
    )
    return o


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        # triton.Config({'BV': BV, 'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        # for BV in [128, 64,]
        # for BK in [128, 64,]
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'USE_G'],
)
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT']),
})
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def chunk_bwd_kernel_qdo_local(
        q,
        g,
        do,
        qdo,
        nats_block_types,
        nats_block_indices,
        cu_seqlens,
        cu_seqlens_nats,
        chunk_indices,
        chunk_indices_op_nats,
        scale,
        T,
        TNAtS,
        H: tl.constexpr,
        HNAtS: tl.constexpr,
        GNAtS: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_G: tl.constexpr,
        NAtS_BLOCK_SIZE: tl.constexpr,
        OFFSET_OP: tl.constexpr,
        N_TYPES: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
        INCOMPLETE_BLOCK_WITH_START_HT: tl.constexpr,
):
    # i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # i_b, i_h = i_bh // H, i_bh % H
    #

    i_v, i_t_, i_gnats = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t_ // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t_ % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t_
        i_t_nats_offset = 0

    off_bh_nats = tl.load(chunk_indices_op_nats + i_t_ * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_op_nats + i_t_ * 2 + 1).to(tl.int32)
    i_b = off_bh_nats // HNAtS
    i_hnats = off_bh_nats % HNAtS

    i_h = i_hnats * GNAtS + i_gnats

    # i_k = i_kv % NBK
    # i_v = i_kv // NBK

    if IS_VARLEN:
        # i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

    # offset calculation
    q += (bos * H + i_h) * K
    do += (bos * H + i_h) * V

    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP
    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    stride_block_types_t = N_TYPES * HNAtS

    stride_kt = K * H
    stride_vt = V * H

    qdo += (i_t_ * GNAtS + i_gnats).to(tl.int64) * K * V

    # TODO any better solution for that?
    # TODO this is only the case for i_t_nats_offset always 0, we need a something different for i_t_nats_offset !=0
    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    i_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
    i_nats_block_last = tl.load(
        nats_block_indices + (load_idx_chunk + (i_t_nats_offset - 1) // N_CHUNK_PER_NAtS_BLOCK) * stride_block_types_t,
        mask=load_idx_chunk > 0, other=0
    ).to(tl.int32)
    i_t0 = i_nats_block_last * NAtS_BLOCK_SIZE + i_t_nats_offset * BT
    # i_t0 = tl.where((i_nats_block==0) & (i_t==0), i_t0 + BT, i_t0, )
    # i_t0 = tl.where((i_t == 0), i_t0, i_t0 + BT, )
    # i_t0 = i_t0 + BT
    # we only compute the non-delta ops
    i_t0 = tl.where(i_t == 0 & (i_nats_block != 0), i_t0, i_t0 + BT)
    # i_t0 = i_t0 + BT
    i_t_next = min(i_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT,
                   T - BT)  # we do not compute the values for nats block is True...

    n_iters = tl.cdiv(i_t_next - i_t0, BT)

    b_qdo1 = tl.zeros([BK, BV], dtype=tl.float32)
    if K > BK:
        b_qdo2 = tl.zeros([BK, BV], dtype=tl.float32)
    if K > 2 * BK:
        b_qdo3 = tl.zeros([BK, BV], dtype=tl.float32)
    if K > 3 * BK:
        b_qdo4 = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_G:
        # since this is a uniform function that is applied for all gated linear attention models, we
        # would assume that g is always stored as full matrix (B, T, H) TODO check if this always hold!
        stride_g = H
        g += bos * H + i_h

    for i in range(n_iters):

        i_tq = i * BT + i_t0

        p_do = tl.make_block_ptr(do, (i_t_next, V), (stride_vt, 1), (i_tq, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_G:
            # nats_load_end = (load_idx_block_start + N_NAtS_BLOCK_PER_T - 1)
            # o_t_nats_last = tl.load(nats_block_indices + nats_load_end* stride_block_types_t,
            #                   mask=nats_load_end < TNAtS, other=TNAtS
            #                   ).to(tl.int32) * N_NAtS_BLOCK_PER_T

            # p_gq = tl.make_block_ptr(gq, (o_t_nats_last,), (H,), (o_qt_start,), (BT,), (0, ))
            p_g = tl.make_block_ptr(g, (i_t_next,), (H,), (i_tq,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g = tl.exp(b_g)

        p_q = tl.make_block_ptr(q, (K, i_t_next), (1, stride_kt), (0, i_tq), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_q = b_q * b_g[None, :]
        b_q = (b_q * scale).to(b_q.dtype)
        b_qdo1 += tl.dot(b_q, b_do.to(b_q.dtype))

        if K > BK:
            p_q = tl.make_block_ptr(q, (K, i_t_next), (1, stride_kt), (BK, i_tq), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_q = b_q * b_g[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_qdo2 += tl.dot(b_q, b_do.to(b_q.dtype))

        if K > 2 * BK:
            p_q = tl.make_block_ptr(q, (K, i_t_next), (1, stride_kt), (2 * BK, i_tq), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_q = b_q * b_g[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_qdo3 += tl.dot(b_q, b_do.to(b_q.dtype))

        if K > 3 * BK:
            p_q = tl.make_block_ptr(q, (K, i_t_next), (1, stride_kt), (3 * BK, i_tq), (BK, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_q = b_q * b_g[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_qdo4 += tl.dot(b_q, b_do.to(b_q.dtype))

    if n_iters > 0:
        p_qdo = tl.make_block_ptr(qdo, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_qdo, b_qdo1.to(p_qdo.dtype.element_ty), boundary_check=(0, 1))

        if K > BK:
            p_qdo = tl.make_block_ptr(qdo, (K, V), (V, 1), (BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_qdo, b_qdo2.to(p_qdo.dtype.element_ty), boundary_check=(0, 1))

        if K > 2 * BK:
            p_qdo = tl.make_block_ptr(qdo, (K, V), (V, 1), (2 * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_qdo, b_qdo3.to(p_qdo.dtype.element_ty), boundary_check=(0, 1))

        if K > 3 * BK:
            p_qdo = tl.make_block_ptr(qdo, (K, V), (V, 1), (3 * BK, i_v * BV), (BK, BV), (1, 0))
            tl.store(p_qdo, b_qdo4.to(p_qdo.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT']),
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'NAtS_BLOCK_SIZE'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqk(
        q,
        k,
        v,
        h,
        g,
        #dt,
        do,
        dh,
        dq,
        dk,
        #ddt,
        dg,
        dnats,
        nats_block_types,
        nats_block_indices,
        n_nats_blocks,
        cu_seqlens,
        cu_seqlens_nats,
        starting_h_idx,
        chunk_indices,
        chunk_indices_op_nats,
        nats_block_op_offsets,
        scale,
        B: tl.constexpr,
        T,
        TNAtS,
        H: tl.constexpr,
        HNAtS: tl.constexpr,
        GNAtS: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_G: tl.constexpr,
        NAtS_BLOCK_SIZE: tl.constexpr,
        OFFSET_OP: tl.constexpr,
        COMPUTE_INCOMPLETE_BLOCK_SCORES: tl.constexpr,
        COMPUTE_DNATS_FOR_INCOMPLETE_SCORES: tl.constexpr,
        N_TYPES: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
):
    # i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # i_b, i_h = i_bh // H, i_bh % H
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t
        i_t_nats_offset = 0

    i_hnats = i_h // GNAtS
    i_gnats = i_h % GNAtS

    start_nats_wvh_t = tl.load(nats_block_op_offsets + (i_b * HNAtS + i_hnats)).to(tl.int32)
    end_nats_wvh_t = tl.load(nats_block_op_offsets + (i_b * HNAtS + i_hnats + 1)).to(tl.int32)

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_b).to(tl.int32), tl.load(cu_seqlens_nats + i_b + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
        starting_h_idx += i_b + i_b * tl.cdiv(T, BT) * HNAtS + i_hnats
        i_th = tl.load(starting_h_idx + i_t * HNAtS).to(tl.int32)
        i_th_last = tl.load(starting_h_idx + (TNAtS - 1) * HNAtS).to(tl.int32)
        ng = T
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

        starting_h_idx += i_b * tl.cdiv(T, BT) * HNAtS + i_hnats
        i_th = tl.load(starting_h_idx + i_t * HNAtS).to(tl.int32)
        i_th_last = tl.load(starting_h_idx + (TNAtS - 1) * HNAtS).to(tl.int32)
        ng = B * T

    # offset calculation
    do += (bos * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K

    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP
    # T_NATS_Compute = tl.load(n_nats_blocks + (i_b * HNAtS + i_hnats) * N_TYPES + OFFSET_OP) * NAtS_BLOCK_SIZE
    dnats += (bos_nats * N_CHUNK_PER_NAtS_BLOCK * H + i_h) + i_k * B * TNAtS * N_CHUNK_PER_NAtS_BLOCK * H

    stride_vt = V * H

    if USE_G:
        dg += i_k * ng * H

        #ddt += i_k * ng * H

        b_dg_last = tl.zeros([1, ], dtype=tl.float32) if USE_G else None
        g += bos * H + i_h
        #dt += bos * H + i_h

        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        
        #p_dt = tl.make_block_ptr(dt, (T,), (H,), (i_t * BT,), (BT,), (0,))
        #b_dt = tl.load(p_dt, boundary_check=(0,))


    h += ((start_nats_wvh_t.to(tl.int64) + i_th) * GNAtS + i_gnats) * K * V
    dh += ((start_nats_wvh_t.to(tl.int64) + i_th) * GNAtS + i_gnats) * K * V

    stride_nats_msk = N_TYPES * H
    stride_kt = K * H
    stride_nats_block = N_TYPES * HNAtS

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    chunk_is_delta = tl.load(nats_block_types + load_idx_chunk * stride_nats_block).to(tl.int1)


    wv_load_shape0 = T
    wv_load_offset = i_t * BT

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_nats_opt = 0.

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    if chunk_is_delta:
        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(v, (wv_load_shape0, V), (stride_vt, 1), (wv_load_offset, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            # [BT, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))

            if USE_G:
                b_dg_last += (tl.sum(b_h * b_dh))
            # [BT, BV] @ [BV, BT] -> [BT, BT]
            if COMPUTE_INCOMPLETE_BLOCK_SCORES:
                b_ds += tl.dot(b_do, tl.trans(b_v))
            # [BT, BV] @ [BV, BK] -> [BT, BK]
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            # [BT, BV] @ [BV, BK] -> [BT, BK]
            b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

        tl.debug_barrier()
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_G:
            b_nats_opt += tl.sum((b_k * b_dk) * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None])
        else:
            b_nats_opt += tl.sum((b_k * b_dk))
        tl.store(dnats + i_t * H, b_nats_opt.to(dnats.dtype.element_ty))
    else:
        # in this case, we only compute dq
        if COMPUTE_DNATS_FOR_INCOMPLETE_SCORES:
            b_dk_virtual = tl.zeros([BT, BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            # [BT, BV]
            # here, we still need to update v to v-wh, However, since computing wh needs to marginalize the K dims,
            # and we only split the K dimension, hence, we compute
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            # [BT, BV] @ [BV, BT] -> [BT, BT]            # [BT, BV] @ [BV, BK] -> [BT, BK]
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            # [BT, BV] @ [BV, BK] -> [BT, BK]

            if COMPUTE_INCOMPLETE_BLOCK_SCORES or COMPUTE_DNATS_FOR_INCOMPLETE_SCORES:
                # We only need to load vk if: we need to compute the incomplete block scores or
                # compute dnats for incomplete scores, otherwise, they will be considered as 0!
                p_v = tl.make_block_ptr(v, (T, V), (stride_vt, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_ds += tl.dot(b_do, tl.trans(b_v))
                if COMPUTE_DNATS_FOR_INCOMPLETE_SCORES:
                    # This is the scores of K as if it is still applied to the output
                    p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                    b_dh = tl.load(p_dh, boundary_check=(0, 1))
                    b_dk_virtual += tl.dot(b_v, b_dh.to(b_v.dtype))

        tl.debug_barrier()
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        if COMPUTE_DNATS_FOR_INCOMPLETE_SCORES:
            if USE_G:
                b_nats_opt += tl.sum((b_dk_virtual * b_k) * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None])
            else:
                b_nats_opt += tl.sum((b_dk_virtual * b_k))
            tl.store(dnats + i_t * H, b_nats_opt.to(dnats.dtype.element_ty))

    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    if COMPUTE_INCOMPLETE_BLOCK_SCORES:
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        if USE_G:
            b_dg = tl.zeros([BT, ], dtype=tl.float32)
            dg += bos * H + i_h
            #ddt += bos * H + i_h
            b_dg_last *= tl.exp(b_g_last)

            b_dq = b_dq * tl.exp(b_g)[:, None] * scale
            b_dg += tl.sum(b_dq * b_q, axis=1)

            b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
            b_dg -= tl.sum(b_k * b_dk, axis=1)
            b_dg_last += tl.sum(b_dk * b_k)
            b_ds = tl.where(m_A, b_ds * tl.exp(b_g[:, None] - b_g[None, :]), 0) * scale
            b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
            b_dg += tl.sum(b_ds2, axis=1)
            b_dg -= tl.sum(b_ds2, axis=0)

            b_ds = b_ds.to(b_k.dtype)
            # [BT, BK]
            b_dq += tl.dot(b_ds, b_k) # * b_dt[:, None]
            b_dk += tl.dot(tl.trans(b_ds), b_q)

            #b_ddt =  tl.sum(b_dk * b_k, axis=1)

            p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
            #p_ddt = tl.make_block_ptr(ddt, (T,), (H,), (i_t * BT,), (BT,), (0,))

            # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
            # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
            b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
            
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            #tl.store(p_dk, (b_dk * b_dt[:, None]).to(p_dk.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dk, (b_dk).to(p_dk.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
            #tl.store(p_ddt, b_ddt.to(p_ddt.dtype.element_ty), boundary_check=(0,))

        else:
            b_ds = tl.where(m_A, b_ds, 0)
            b_ds = b_ds.to(b_k.dtype)
            b_dq += tl.dot(b_ds, b_k)
            b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
            b_dq *= scale
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    else:
        # in this case, we will not compute any b_s related items since they will not be computed
        if USE_G:
            b_dg = tl.zeros([BT, ], dtype=tl.float32)
            dg += bos * H + i_h
            #ddt += bos * H + i_h
            b_dg_last *= tl.exp(b_g_last)
            b_dq = b_dq * tl.exp(b_g)[:, None] * scale
            b_dg += tl.sum(b_dq * b_q, axis=1)

            if chunk_is_delta:
                b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
                b_dg -= tl.sum(b_k * b_dk, axis=1)
                b_dg_last += tl.sum(b_dk * b_k)
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            #b_ddt =  tl.sum(b_dk * b_k, axis=1)
            #p_ddt = tl.make_block_ptr(ddt, (T,), (H,), (i_t * BT,), (BT,), (0,))
            #tl.store(p_ddt, b_ddt.to(p_ddt.dtype.element_ty), boundary_check=(0,))
            
            p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
            # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
            # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
            b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

        else:
            tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
            if chunk_is_delta:
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

def chunk_bwd_nats_dqkwg(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        h: torch.Tensor,
        dh: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        nats_block_op_offsets: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        starting_h_idx: torch.Tensor,
        chunk_indices_op_nats: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        #dt: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        scale: float = 1.0,
        nats_block_size: int = 64,
        offset_op: int = 0,
        compute_incomplete_block_scores: bool = False,
        compute_dnats_for_invalid_blocks: bool = True,
        decay_for_non_gdn_blocks:bool=False,
        incomplete_block_start_with_ht: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS

    wv_are_flattened = not (compute_incomplete_block_scores or compute_dnats_for_invalid_blocks)

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(triton.next_power_of_2(K), CONST_TILING)
    BV = min(triton.next_power_of_2(V), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k)

    dg = torch.zeros(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    #ddt = torch.zeros(NK, *dt.shape, dtype=torch.float32, device=g.device) if dt is not None else None
    dnats = torch.zeros(NK, B, TNAtS * triton.cdiv(nats_block_size, BT), H, dtype=k.dtype, device=k.device)

    grid = (NK, NT, B * H)
    # grid = (NK, len(chunk_indices_op_nats), GNAtS)
    chunk_bwd_kernel_dqk[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        #dt=dt,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        #ddt=ddt,
        dg=dg,
        dnats=dnats,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        nats_block_op_offsets=nats_block_op_offsets,
        n_nats_blocks=n_nats_blocks,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        starting_h_idx=starting_h_idx,
        chunk_indices=chunk_indices,
        chunk_indices_op_nats=chunk_indices_op_nats,
        scale=scale,
        B=B,
        T=T,
        TNAtS=TNAtS,
        H=H,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NAtS_BLOCK_SIZE=nats_block_size,
        N_TYPES=n_opts,
        OFFSET_OP=offset_op,
        COMPUTE_INCOMPLETE_BLOCK_SCORES=compute_incomplete_block_scores,
        COMPUTE_DNATS_FOR_INCOMPLETE_SCORES=compute_dnats_for_invalid_blocks,
    )
    if dg is not None:
        dg = dg.sum(0)
        #ddt = ddt.sum(0)
    if nats_block_size > BT:
        dnats = reduce(dnats, 'k b (t gt) (h g) -> b t h',
                       g=GNAtS, gt=triton.cdiv(nats_block_size, BT), reduction='sum')
    else:
        dnats = reduce(dnats, 'k b t (h g) -> b t h',
                       g=GNAtS, reduction='sum')

    return dq, dk, dg, dnats


def chunk_bwd_dv_qdo_nats_local(
        q: torch.Tensor,
        k: torch.Tensor,
        do: torch.Tensor,
        v: Optional[torch.Tensor],
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        chunk_indices_op_nats: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        scale: float = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        nats_block_size: int = 8,
        offset_op: int = 0,
        compute_incomplete_block_scores: bool = True,
        pre_compute_qdo: bool = True,  # TODO check if we really need this!
        incomplete_block_start_with_ht: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if compute_incomplete_block_scores:
        dv = chunk_bwd_dv_local(q, k, do, g, None, scale, cu_seqlens, chunk_size)
    else:
        dv = None
    if pre_compute_qdo:
        # This function is applied to cumulatively comptue both dv and do.T@ q that ends at the tail of each
        # computation chunk of BT that only contains the valid kv nats blocks
        B, T, H, K, V = *k.shape, do.shape[-1]
        B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
        GNAtS = H // HNAtS

        BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
        # H100 can have larger block size
        """
        if check_shared_mem('hopper', k.device.index):
            CONST_TILING = 128
        elif check_shared_mem:
            CONST_TILING = 64
        else:
            CONST_TILING = 32
        """
        if check_shared_mem('hopper', k.device.index):
            CONST_TILING = 128
        else:
            CONST_TILING = 64

        BK = min(triton.next_power_of_2(K), CONST_TILING)
        BV = min(triton.next_power_of_2(V), CONST_TILING)
        NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

        N_TYPES = nats_block_types.shape[-1]
        # grid = (len(chunk_indices_op_nats), GNAtS, )
        # grid = (len(chunk_indices_op_nats), GNAtS)
        # dv = torch.empty_like(do)
        def grid(meta):
            return (triton.cdiv(V, meta['BV']), len(chunk_indices_op_nats), GNAtS)

        # for qdo, we need something different: if the last nats block in  the row is actually not the last
        # computational chunk, we store them to the next h. Hence, for any chunk_indices, if its next block indices
        # is TNAtS, we only do one iteration and stll compute the remaining parts. Hence, we will add the remaining index
        # here
        qdo = torch.zeros(len(chunk_indices_op_nats), GNAtS, K, V, device=do.device, dtype=do.dtype)
        BK = 64
        BV = 64
        # The usage of this function is three-fold, we compute the cumulative q@do- dv_local for the non-delta
        # operations, Additionally, we also update v_new for non-nats block: v_new = v - w@h as these v values are not
        # updated in the h computations

        chunk_bwd_kernel_qdo_local[grid](
            q=q,
            g=g,
            do=do,
            qdo=qdo,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            chunk_indices=chunk_indices,
            chunk_indices_op_nats=chunk_indices_op_nats,
            scale=scale,
            T=T,
            TNAtS=TNAtS,
            H=H,
            HNAtS=HNAtS,
            GNAtS=GNAtS,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            NAtS_BLOCK_SIZE=nats_block_size,
            N_TYPES=n_opts,
            OFFSET_OP=offset_op,
            INCOMPLETE_BLOCK_WITH_START_HT=incomplete_block_start_with_ht,
        )
    else:
        qdo = None

    return dv, qdo

