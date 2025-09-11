# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import copy
from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

from nats.ops.nats_util import prepare_nats_block_indices



@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT'])
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'BT', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    nats_block_types,
    nats_block_indices, # both are of shape [B,T, HAttns]
    cu_seqlens,
    cu_seqlens_nats,
    chunk_indices,
    chunk_indices_op_nats,
    T,
    TNAtS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    HNAtS: tl.constexpr,
    GNAtS: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NAtS_BLOCK_SIZE: tl.constexpr,
    N_TYPES: tl.constexpr,
    OFFSET_OP: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
):
    i_t_, i_gnats = tl.program_id(0), tl.program_id(1)
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

    if IS_VARLEN:
        #i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

    beta += bos * H + i_h
    k += (bos * H + i_h) * K

    A += (i_t_ * BT * GNAtS + i_gnats).to(tl.int64) * BT

    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    stride_block_types_t = N_TYPES * HNAtS
    stride_kt = K * H
    stride_At = BT * GNAtS

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t)
    i_t0 = b_o_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t0,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (stride_kt, 1), (i_t0, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t0,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= tl.exp(b_g_diff)
    b_A *= b_beta[:, None]

    o_t = i_t0 + tl.arange(0, BT)
    m_t = o_t < T

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A, (T - i_t0, BT), (stride_At, 1), (0, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"]
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        b_kt = tl.make_block_ptr(k, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        # [BK,]
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H*K + o_k, mask=m_k, other=0)
        # [BC, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        # [BK, BC]
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1)) * exp(b_gn[:, None] - b_gk)
        # [BC, BC]
        b_A += tl.dot(b_k, b_kt)
    b_A *= b_beta[:, None]

    p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BT"]
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H*BT + i_h * BT + i_i * BC

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_beta = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h

    b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_beta, mask=m_A, other=0)[:, None]
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_kt = k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_k * b_kt[None, :] * exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A, 0.)

        tl.store(A + o_A + j, b_A, mask=m_A)
        p_kt += H*K
        p_gk += H*K



def chunk_scaled_dot_kkt_nats_fwd(
    k: torch.Tensor,
    nats_block_types: torch.Tensor,
    nats_block_indices: torch.Tensor,
    n_nats_blocks: torch.Tensor,
    chunk_indices_op_nats: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_nats: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    nats_block_size: int = 8,
    offset_op: int = 0,
    output_dtype: torch.dtype = torch.float32,
    keep_wu_as_kv: bool=False,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        nats_block_types (torch.Tensor):
            The types of each nats chunk of shape `[B, TNAtS, HNAtS]`
        nats_block_indices (torch.Tensor):
            index of each nats chunks of shape `[B, TNAtS, HNAtS]`, help us to quickly load the corresponing kv values
        offset_op (torch.Tensors):
            number of nats chunks for each options of shape [B, HNATs, n_opts]
        chunk_indices_op_nats (torch.Tensor):
            we flatten the output tensors to only store the values of the delta blocks. Hence, different heads might
            have different delta blocks. This value is used to record the indices of each BT chunk (similar to
            chunk_indices in the var_len cases)
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    if keep_wu_as_kv:
        return chunk_scaled_dot_kkt_fwd(
            k=k,
            g=g,
            gk=gk,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            output_dtype=output_dtype
        )
    B, T, H, K = k.shape
    BT = chunk_size
    B, TNAtS, HNAtS, n_opts = nats_block_indices.shape
    # TODO check if we actually need to check this cumsum for every operations?
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    GNAtS = H // HNAtS

    #chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    #chunk_indices_op_nats = prepare_nats_block_indices(n_nats_blocks, nats_block_size, BT, offset_delta)
    #n_nats_blocks_sum = torch.sum(n_nats_blocks[..., offset_delta])
    #cu_seqlens_nats = torch.cumsum(n_nats_blocks[..., offset_delta], 0)
    #A = torch.zeros(cu_seqlens_nats[-1], HNAtS, BT, device=k.device, dtype=output_dtype)
    # TODO This requires a bit more memories but easier to be computed?
    A = torch.zeros(len(chunk_indices_op_nats) * BT, GNAtS, BT, device=k.device, dtype=output_dtype)
    N_TYPES = nats_block_types.shape[-1]
    if gk is None:
        #NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
        #A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
        grid = (len(chunk_indices_op_nats), GNAtS)

        chunk_scaled_dot_kkt_fwd_kernel[grid](
            k=k,
            g=g,
            beta=beta,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            chunk_indices_op_nats=chunk_indices_op_nats,
            A=A,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            chunk_indices=chunk_indices,
            T=T,
            TNAtS=TNAtS,
            NAtS_BLOCK_SIZE=nats_block_size,
            GNAtS=GNAtS,
            N_TYPES=N_TYPES,
            H=H,
            HNAtS=HNAtS,
            K=K,
            BT=BT,
            OFFSET_OP=offset_op,
        )
        return A
    raise NotImplementedError("gk is not implemnted for NAtS!")

    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    # TODO finish the followings
    grid = (len(chunk_indices_op_nats), NC * NC, GNAtS)

    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
        k=k,
        g=gk,
        beta=beta,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        chunk_indices_op_nats=chunk_indices_op_nats,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        TNAtS=TNAtS,
        NAtS_BLOCK_SIZE=nats_block_size,
        GNAtS=GNAtS,
        H=H,
        HNAtS=HNAtS,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        OFFSET_OP=offset_delta,
    )

    grid = (len(chunk_indices_op_nats), NC, GNAtS)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        k=k,
        g=gk,
        beta=beta,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        chunk_indices_op_nats=chunk_indices_op_nats,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        TNAtS=TNAtS,
        NAtS_BLOCK_SIZE=nats_block_size,
        GNAtS=GNAtS,
        H=H,
        HNAtS=HNAtS,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        OFFSET_OP=offset_delta,
    )
    return A


def test_scaled_dot_kkt_fwd(dtype=torch.bfloat16):
    torch.manual_seed(0)
    from nats.utils import check_fp16_dtype
    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16
    #dtype = torch.float16

    B = 2
    H = 4
    HNatS = 4
    T = 512
    N_TYPES = 3
    GNAtS = H // HNatS
    delta_offset = 1
    K = 128
    NATS_Chunk = 8
    chunk_size = 64
    T_NAtS = triton.cdiv(T, NATS_Chunk)
    device = torch.device('cuda')
    k = torch.randn(B, T, H, K, dtype=dtype, device=torch.device('cuda'))
    beta = torch.randn(B, T, H, dtype=dtype, device=device)
    g = torch.randn(B, T, H, dtype=torch.float32, device=device)

    logits = torch.randn(B, T_NAtS, HNatS, N_TYPES, device=torch.device('cuda'), dtype=dtype)
    nats_types = torch.nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
    # attN_TYPES[...,0] =1.
    nats_idx = torch.where(nats_types == 1., torch.arange(T_NAtS, device=nats_types.device).view(1, -1, 1, 1), T_NAtS)
    nats_idx = nats_idx.sort(1)[0]
    n_nats_blocks = torch.sum(nats_types.long(), dim=1)

    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

    mask_ = nats_types[..., delta_offset]
    mask_ = mask_.repeat_interleave(NATS_Chunk, 1)
    mask_ = mask_[:,:,:,None].expand(B, NATS_Chunk * T_NAtS, HNatS, GNAtS).reshape(B, NATS_Chunk * T_NAtS, HNatS * GNAtS)
    mask_ = mask_[:,:T, :]
    k1 = copy.deepcopy(k) * mask_[..., None]

    #out1 = chunk_scaled_dot_kkt_fwd(k1, beta=beta, g=g, chunk_size=chunk_size)

    out2 = chunk_scaled_dot_kkt_nats_fwd(
        k=k,
        nats_block_types=nats_types,
        nats_block_indices=nats_idx,
        n_nats_blocks=n_nats_blocks,
        beta=beta,
        g=g,
        offset_delta=delta_offset,
        nats_block_size=NATS_Chunk,
        chunk_size=chunk_size,
    )
    i_start = 0
    mask_ = mask_.bool()
    torch.set_printoptions(sci_mode=False)
    for b in range(B):
        for h in range(H):
            h_nats = h // GNAtS

            n_nats_block = n_nats_blocks[b, h_nats, delta_offset]
            i_end = i_start + triton.cdiv(n_nats_block * NATS_Chunk, chunk_size) * chunk_size
            out_nats = out2[i_start:i_start + n_nats_block * NATS_Chunk, h % GNAtS]

            out_naive = out1[b,:, h][mask_[b,:, h]]

            diff = ((out_nats).sort(0)[0] - out_naive.sort(0)[0])

            print(f'b: {b}, h: {h}, min diff: {diff.min()}')
            print(f'b: {b}, h: {h}, max diff: {diff.max()}')
            print(f"out1 remains: {out1[b,:, h][~mask_[b,:, h]].abs().sum()}")
            print(f"out2 remains: {out2[i_start + n_nats_chunk * NATS_Chunk: i_end, h % GNAtS].abs().sum()}")
            i_start = i_end

            # TODO find a proper way to check if this is correct:


            import pdb
            pdb.set_trace()



if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    test_scaled_dot_kkt_fwd()









