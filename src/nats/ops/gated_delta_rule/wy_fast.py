# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from nats.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_nats_fwd
# from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from nats.ops.utils.solve_tril import solve_tril_nats
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem, is_nvidia_hopper

from nats.ops.nats_util import prepare_nats_block_indices

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT'])
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'HNAtS', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def recompute_w_u_nats_fwd_kernel(
        k,
        v,
        beta,
        w,
        u,
        A,
        g,
        gk,
        nats_block_types,
        nats_block_indices,
        cu_seqlens,
        cu_seqlens_nats,
        chunk_indices,
        chunk_indices_delta_nats,
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
        USE_GK: tl.constexpr,
        NAtS_BLOCK_SIZE: tl.constexpr,
        N_TYPES: tl.constexpr,
        OFFSET_DELTA: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
):
    # i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # i_b, i_h = i_bh // H, i_bh % H

    i_t_, i_gnats = tl.program_id(0), tl.program_id(1)

    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t_ // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t_ % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t_
        i_t_nats_offset = 0

    off_bh_nats = tl.load(chunk_indices_delta_nats + i_t_ * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_delta_nats + i_t_ * 2 + 1).to(tl.int32)
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
    # for k,v and beta we need to load them according to the indices given by nats_block_indices
    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_DELTA

    stride_block_types_t = N_TYPES * HNAtS

    beta += bos * H + i_h
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    # for w, u, v, we need to load according to the indices given by i_t_
    A += (i_t_ * BT * GNAtS + i_gnats).to(tl.int64) * BT
    w += (i_t_ * BT * GNAtS + i_gnats).to(tl.int64) * K
    u += (i_t_ * BT * GNAtS + i_gnats).to(tl.int64) * V

    stride_wt = GNAtS * K
    stride_ut = GNAtS * V
    stride_At = GNAtS * BT

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
    i_t0 = b_o_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t0,), (BT,), (0,))
    p_A = tl.make_block_ptr(A, (T - i_t0, BT), (stride_At, 1), (0, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u, (T - i_t0, V), (stride_ut, 1), (0, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t0,), (BT,), (0,))
        b_g = exp(tl.load(p_g, boundary_check=(0,)))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t0, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w, (T - i_t0, K), (stride_wt, 1), (0, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        if USE_G:
            b_kb *= b_g[:, None]
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t0, i_k * BK), (BT, BK), (1, 0))
            b_kb *= exp(tl.load(p_gk, boundary_check=(0, 1)))

        b_w = tl.dot(b_A.to(b_kb.dtype), b_kb, allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'HNAtS', 'IS_VARLEN'],
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT'])
})
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_nats_kernel(
        k,
        v,
        beta,
        g,
        A,
        dw,
        du,
        dk,
        dv,
        dbeta,
        dg,
        nats_block_types,
        nats_block_indices,
        cu_seqlens,
        cu_seqlens_nats,
        chunk_indices,
        chunk_indices_delta_nats,
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
        OFFSET_DELTA: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
        KEEP_WU_AS_KV: tl.constexpr,
):
    # i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # i_b, i_h = i_bh // H, i_bh % H
    i_t_, i_gnats = tl.program_id(0), tl.program_id(1)
    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t_ // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t_ % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t_
        i_t_nats_offset = 0

    off_bh_nats = tl.load(chunk_indices_delta_nats + i_t_ * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_delta_nats + i_t_ * 2 + 1).to(tl.int32)
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

    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_DELTA
    stride_block_types_t = N_TYPES * HNAtS

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
    i_t0 = b_o_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT
    
    if KEEP_WU_AS_KV:
        dw += (bos * H + i_h).to(tl.int64) * K
        du += (bos * H + i_h).to(tl.int64) * V
        A += (bos*H + i_h).to(tl.int64) * BT

        stride_wt = H * K
        stride_ut = H * V
        p_A = tl.make_block_ptr(A,  (BT, T), (1, H*BT), (0, i_t0), (BT, BT), (0, 1))

        wu_load_shape0 = T
        wu_load_offset = i_t0
        
    else:
        dw += (i_t_ * BT * GNAtS + i_gnats).to(tl.int64) * K
        du += (i_t_ * BT * GNAtS + i_gnats).to(tl.int64) * V + i_gnats * V
        A += i_t_.to(tl.int64) * BT * GNAtS * BT + i_gnats * BT

        stride_wt = GNAtS * K
        stride_ut = GNAtS * V
        p_A = tl.make_block_ptr(A,  (BT, T - i_t0), (1, GNAtS * BT), (0, 0), (BT, BT), (0, 1))
        
        wu_load_shape0 = T - i_t0
        wu_load_offset = 0
        

    p_beta = tl.make_block_ptr(beta + (bos * H + i_h), (T,), (H,), (i_t0,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t0,), (BT,), (0,))
    # p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_g_exp = tl.exp(b_g)

    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t0, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t0, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw, (wu_load_shape0, K), (stride_wt, 1), (wu_load_offset, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k_beta_g = (b_k * b_beta[:, None] * b_g_exp[:, None]).to(b_k.dtype)
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_k_beta_g))
        b_dk_beta_g = tl.dot(b_A, b_dw)
        b_dk = b_dk_beta_g * b_beta[:, None] * b_g_exp[:, None]
        b_dbeta += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None], 1)
        b_dg += tl.sum(b_dk_beta_g * b_k * b_g_exp[:, None] * b_beta[:, None], 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du, (wu_load_shape0, V), (stride_ut, 1), (wu_load_offset, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_v_beta))
        b_dv_beta = tl.dot(b_A, b_du)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA * exp(b_g[:, None] - b_g[None, :]), 0)
    b_dA = b_dA.to(k.dtype.element_ty)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t0, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t0, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_A += tl.dot(b_k_beta, tl.trans(b_k))
        b_dk_beta = tl.dot(b_dA, b_k)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += tl.dot(tl.trans(b_dA), b_k_beta)
        b_dk += b_dk_beta * b_beta[:, None]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    b_dA_A = b_dA * b_A
    b_dg += tl.sum(b_dA_A, axis=1) - tl.sum(b_dA_A, axis=0)
    p_dg = tl.make_block_ptr(dg + (bos * H + i_h), (T,), (H,), (i_t0,), (BT,), (0,))
    p_dbeta = tl.make_block_ptr(dbeta + (bos * H + i_h), (T,), (H,), (i_t0,), (BT,), (0,))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,))


def prepare_wy_repr_nats_fwd(
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor],
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        cu_seqlens_delta: Optional[torch.LongTensor] = None,
        nats_block_size: int = 1,
        offset_delta: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_indices_delta_nats = prepare_nats_block_indices(n_nats_blocks,
                                                          nats_block_size,
                                                          chunk_size,
                                                          offset_delta)
    A = chunk_scaled_dot_kkt_nats_fwd(
        k=k,
        beta=beta,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        g=g,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        offset_delta=offset_delta,
        output_dtype=torch.float32,
    )
    A = solve_tril_nats(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    w, u = recompute_w_u_nats_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
    )
    return w, u, A


def recompute_w_u_nats_fwd(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        chunk_indices_delta_nats: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        gk: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.Tensor] = None,
        nats_block_size: int = 8,
        offset_delta: int = 0,
        keep_wu_as_kv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if keep_wu_as_kv:
        return recompute_w_u_fwd(k=k,
                                 v=v,
                                 beta=beta,
                                 A=A,
                                 g=g,
                                 gk=gk,
                                 cu_seqlens=cu_seqlens)
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(triton.next_power_of_2(K), CONST_TILING)
    BV = min(triton.next_power_of_2(V), CONST_TILING)

    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (len(chunk_indices_delta_nats), GNAtS)

    # u = torch.empty_like(v)
    # w = torch.empty_like(k)
    u = torch.empty(len(chunk_indices_delta_nats) * BT, GNAtS, V, device=v.device, dtype=v.dtype)
    w = torch.empty(len(chunk_indices_delta_nats) * BT, GNAtS, K, device=k.device, dtype=k.dtype)
    recompute_w_u_nats_fwd_kernel[grid](
        k,
        v,
        beta,
        w,
        u,
        A,
        g=g,
        gk=gk,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_indices=chunk_indices,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
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
        OFFSET_DELTA=offset_delta,
    )
    return w, u

def prepare_wy_repr_nats_bwd(
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        dw: torch.Tensor,
        du: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        chunk_indices_delta_nats: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor],
        cu_seqlens_nats:  Optional[torch.LongTensor]=None,
        nats_block_size:int=64,
        offset_delta:int=1,
        compute_incomplete_chunk_scores:bool=True,
        keep_wu_as_kv: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # if we do not compute incomplete chunk scores, then there is no grad for
    # dk, dv, dbeta, dg for invalid blocks
    if compute_incomplete_chunk_scores:
        return prepare_wy_repr_bwd(k, v, g, beta, A, dw, du, cu_seqlens)

    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS

    grid = (len(chunk_indices_delta_nats), GNAtS)

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    dg = torch.zeros_like(g)

    prepare_wy_repr_bwd_nats_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        dbeta=dbeta,
        dg=dg,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_indices=chunk_indices,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        T=T,
        TNAtS=TNAtS,
        H=H,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        N_TYPES=nats_block_types.shape[-1],
        OFFSET_DELTA=offset_delta,
        K=K,
        V=V,
        NAtS_BLOCK_SIZE=nats_block_size,
        BT=BT,
        BK=BK,
        BV=BV,
        KEEP_WU_AS_KV=keep_wu_as_kv,
    )
    return dk, dv, dbeta, dg


fwd_prepare_wy_nats_repr = prepare_wy_repr_nats_fwd

bwd_prepare_wy_nats_repr = prepare_wy_repr_nats_bwd

fwd_recompute_nats_w_u = recompute_w_u_nats_fwd
