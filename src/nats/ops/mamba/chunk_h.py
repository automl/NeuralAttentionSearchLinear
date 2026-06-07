# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp
from fla.utils import check_shared_mem, use_cuda_graph

BKV_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'USE_G': lambda args: args['dA'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT'])
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    dA,
    h0,
    ht,
    nats_block_types,
    nats_block_indices,
    n_nats_blocks,
    cu_seqlens,
    cu_seqlens_nats,
    chunk_offsets,
    nats_block_mamba_offsets,
    B,
    T,
    TNAtS,
    H: tl.constexpr,
    HNAtS: tl.constexpr,
    GNAtS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NAtS_BLOCK_SIZE: tl.constexpr,
    N_TYPES: tl.constexpr,
    OFFSET_H: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DECAY_FOR_NON_LATN_BLOCKS:tl.constexpr,
    N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    i_hnats = i_h // GNAtS
    i_gnats = i_h % GNAtS

    start_nats_wvh_t = tl.load(nats_block_mamba_offsets + (i_n * HNAtS + i_hnats)).to(tl.int32)
    end_nats_wvh_t = tl.load(nats_block_mamba_offsets + (i_n * HNAtS + i_hnats + 1)).to(tl.int32)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        #NT = tl.cdiv(T, BT)
        #boh = tl.load(chunk_offsets + i_n).to(tl.int32)

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        bos, eos = i_n * T, i_n * T + T
        #NT = tl.cdiv(T, BT)
        #boh = i_n * NT

        bos_nats, eos_nats = i_n * TNAtS, i_n * TNAtS + TNAtS

    n_iters = end_nats_wvh_t - start_nats_wvh_t
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    stride_k = H * K
    stride_v = H * V

    stride_h = GNAtS * K * V
    stride_nats_block = N_TYPES * HNAtS

    h += (start_nats_wvh_t.to(tl.int64) * GNAtS + i_gnats) * K * V
    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_H
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_H

    if USE_G:
        stride_g = H
        dA += bos * H + i_h
        #dt += bos * H + i_h
    
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V
    
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 = tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(n_iters):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
        
        load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
        b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_nats_block).to(tl.int32)
        i_t_nats_offset = i_t % N_CHUNK_PER_NAtS_BLOCK
        i_t0 = b_o_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT
        
        #p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, wv_load_offset), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))

        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min(i_t0 + BT, T) - 1

        # scalar decay
        if USE_G:
            o_t = i_t0 + tl.arange(0, BT)
            m_t = o_t < T

            b_g_last = tl.load(dA + last_idx * stride_g)

            p_g = tl.make_block_ptr(dA, (T,), (stride_g,), (i_t0,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            
            #p_dt = tl.make_block_ptr(dt, (T,), (stride_g,), (i_t0,), (BT,), (0,))
            #b_dt = tl.load(p_dt, boundary_check=(0,))

            scale = tl.exp(tl.minimum((b_g_last - b_g), 0.0)) # * b_dt
            b_v = (b_v * scale[:, None]).to(b_v.dtype)
            b_g_last_exp = tl.exp(b_g_last)
            b_h1 *= b_g_last_exp
            if K > 64:
                b_h2 *= b_g_last_exp
            if K > 128:
                b_h3 *= b_g_last_exp
            if K > 192:
                b_h4 *= b_g_last_exp

        # [BK, BT]
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t0), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)

        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t0), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t0), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t0), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

def chunk_mamaba_nats_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    dA: Optional[torch.Tensor],
    nats_block_types: torch.Tensor,
    nats_block_indices: torch.Tensor,
    n_nats_blocks: torch.Tensor,
    nats_block_mamba_offsets: torch.Tensor,
    chunk_indices_mamba_nats: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_nats: Optional[torch.LongTensor] = None,
    nats_block_size: int = 8,
    offset_mamba: int = 1,
    compute_incomplete_chunk_scores:bool = False,
    incomplete_block_start_with_ht:bool = True,
    decay_for_non_mamba_blocks: bool = False,):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    #

    #h = k.new_empty(B, NT, H, K, V)
    #h = torch.empty(len(chunk_indices_mamba_nats), GNAtS, K, V, device=k.device, dtype=k.dtype)
    h = torch.empty(len(chunk_indices_mamba_nats), GNAtS, K, V, device=k.device, dtype=k.dtype)

    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        dA=dA,
        h=h,
        h0=initial_state,
        ht=final_state,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_offsets=chunk_offsets,
        nats_block_mamba_offsets=nats_block_mamba_offsets,
        B=B,
        T=T,
        TNAtS=TNAtS,
        H=H,
        K=K,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        V=V,
        BT=BT,
        NAtS_BLOCK_SIZE=nats_block_size,
        N_TYPES=n_opts,
        OFFSET_H=offset_mamba,
        DECAY_FOR_NON_LATN_BLOCKS=decay_for_non_mamba_blocks,
    )
    return h, final_state




@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT'])
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [4, 3, 2]
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def chunk_mamba_rule_bwd_kernel_dhv_blockdim64(
    q,
    k,
    g,
    dht,
    dh0,
    do,
    dh,
    qdo,
    dv,
    dv2,
    nats_block_types,
    nats_block_indices,
    n_nats_blocks,
    cu_seqlens,
    cu_seqlens_nats,
    chunk_offsets,
    nats_block_mamba_offsets,
    scale,
    T,
    TNAtS,
    H: tl.constexpr,
    K: tl.constexpr,
    HNAtS: tl.constexpr,
    GNAtS: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NAtS_BLOCK_SIZE: tl.constexpr,
    N_TYPES: tl.constexpr,
    OFFSET_MAMBA: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    i_hnats = i_h // GNAtS
    i_gnats = i_h % GNAtS

    start_nats_wvh_t = tl.load(nats_block_mamba_offsets + (i_n * HNAtS + i_hnats)).to(tl.int32)
    end_nats_wvh_t = tl.load(nats_block_mamba_offsets + (i_n * HNAtS + i_hnats + 1)).to(tl.int32)

    #start_nats_h_t = tl.load(nats_block_mamba_offsets + 1 + 2 * (i_n * HNAtS + i_hnats)).to(tl.int32)
    #end_nats_h_t = tl.load(nats_block_mamba_offsets + 1 + 2 * (i_n * HNAtS + i_hnats + 1)).to(tl.int32)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

        bos_nats, eos_nats = i_n * TNAtS, i_n * TNAtS + TNAtS

    # [BK, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    n_iters = end_nats_wvh_t - start_nats_wvh_t

    if USE_G:
        stride_g = H
        g += bos * H + i_h

    qdo += (start_nats_wvh_t.to(tl.int64) * GNAtS + i_gnats) * K * V
    dh += (start_nats_wvh_t.to(tl.int64) * GNAtS + i_gnats) * K * V
    #dv2 += ((start_nats_wvh_t * BT).to(tl.int64) * GNAtS + i_gnats) * V

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V

    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_MAMBA
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_MAMBA
    T_NATS_Compute = tl.load(n_nats_blocks + (i_n * HNAtS + i_hnats) * N_TYPES + OFFSET_MAMBA) * NAtS_BLOCK_SIZE

    stride_h = GNAtS * K * V
    stride_k = H * K
    stride_v = H * V
    stride_do = H * V
    stride_nats_block = N_TYPES * HNAtS

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(n_iters - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t*stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
        b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_nats_block).to(tl.int32)

        i_t_nats_offset = i_t % N_CHUNK_PER_NAtS_BLOCK
        i_t0 = b_o_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT

        if USE_G:
            last_idx = min(i_t0 + BT, T) - 1
            bg_last = tl.load(g + last_idx * stride_g)
            bg_last_exp = tl.exp(bg_last)
            p_g = tl.make_block_ptr(g, (T,), (stride_g,), (i_t0,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = tl.exp(b_g)
        else:
            bg_last = None
            last_idx = None
            b_g = None
            b_g_exp = None
        
        p_dv = tl.make_block_ptr(dv,  (T, V), (stride_v, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t0, 0), (BT, 64), (1,0))
        #b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        #b_dv = tl.load(p_dv, boundary_check=(0,1))
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        # Update dv

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t0, 0), (BT, 64), (1,0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t0, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t0, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t0, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t0 + tl.arange(0, BT)) < T
            b_dv *= tl.where(m_t, tl.exp(bg_last - b_g), 0)[:, None] 
        b_dv += tl.load(p_dv, boundary_check=(0,1))
        p_dv2 = tl.make_block_ptr(dv2,  (T, V), (stride_v, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # Update dh
        p_qdo = tl.make_block_ptr(qdo + i_t*stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_do, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))
        b_qdo = tl.load(p_qdo, boundary_check=(0, 1))

        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t0), (64, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0,1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        b_q = (b_q * scale).to(b_q.dtype)

        b_dh1 += b_qdo + tl.dot(b_q, b_do.to(b_q.dtype))

        if K > 64:
            p_qdo = tl.make_block_ptr(qdo + i_t * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_qdo = tl.load(p_qdo, boundary_check=(0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t0), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh2 += (b_qdo + tl.dot(b_q, b_do.to(b_q.dtype)))
        if K > 128:
            p_qdo = tl.make_block_ptr(qdo + i_t * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_qdo = tl.load(p_qdo, boundary_check=(0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t0), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh3 += (b_qdo + tl.dot(b_q, b_do.to(b_q.dtype)))
        if K > 192:
            p_qdo = tl.make_block_ptr(qdo + i_t * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_qdo = tl.load(p_qdo, boundary_check=(0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t0), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh4 += (b_qdo + tl.dot(b_q, b_do.to(b_q.dtype)))


    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))



def chunk_mamba_nats_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    dht: Optional[torch.Tensor],
    do: torch.Tensor,
    dv: torch.Tensor,
    qdo: torch.Tensor,
    nats_block_types: torch.Tensor,
    nats_block_indices: torch.Tensor,
    n_nats_blocks: torch.Tensor,
    chunk_indices_mamba_nats: torch.Tensor,
    nats_block_mamba_offsets: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_nats: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    nats_block_size: int = 64,
    OFFSET_MAMBA: int = 1,
    compute_incomplete_chunk_scores: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64

    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    GNAtS = H // HNAtS

    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = torch.empty(len(chunk_indices_mamba_nats), GNAtS, K, V, device=q.device, dtype=q.dtype)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None

    #dv2 = torch.empty(len(chunk_indices_mamba_nats) * BT, GNAtS, V, device=do.device, dtype=do.dtype)
    #dv2 = torch.empty_like(dv)
    dv2 = dv.clone()
    #def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)

    chunk_mamba_rule_bwd_kernel_dhv_blockdim64[grid](
        q=q,
        k=k,
        g=g,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        qdo=qdo,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_offsets=chunk_offsets,
        nats_block_mamba_offsets=nats_block_mamba_offsets,
        scale=scale,
        T=T,
        TNAtS=TNAtS,
        H=H,
        K=K,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        V=V,
        BT=BT,
        NAtS_BLOCK_SIZE=nats_block_size,
        N_TYPES=n_opts,
        OFFSET_MAMBA=OFFSET_MAMBA,
    )

    return dh, dh0, dv2
