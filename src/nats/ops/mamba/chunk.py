# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
# from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from nats.ops.utils.cumsum import chunk_cumsum_non_gated_chunks
from fla.ops.utils import chunk_local_cumsum
from nats.ops.utils import solve_tril_nats
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks
from nats.ops.mamba.chunk_h import chunk_mamaba_nats_fwd_h, chunk_mamba_nats_bwd_dhu
from nats.ops.mamba.chunk_o import chunk_mamba_fwd_nats_o, chunk_bwd_dv_qdo_nats_local, chunk_bwd_nats_dqkwg
from nats.ops.mamba.chunk_cumsum import chunk_nats_cumsum_fwd, chunk_nats_cumsum_bwd

def chunk_mamba2_fwd(
        X: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor,
        g: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor, 
        n_nats_blocks: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        chunk_indices_mamba_nats: torch.Tensor,
        nats_block_mamba_offsets: torch.Tensor,
        starting_h_idx_mamba: torch.Tensor,
        dt_softplus: bool = True,
        decay_for_non_mamba_blocks: bool= True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        nats_block_size: int = 64,
        offset_mamba: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        incomplete_block_start_with_ht: bool = True,
        chunk_size: int = 64,
        dt_limit=(0.0, float("inf")),
):
    assert nats_block_size >= chunk_size, "The current implementaion only allows one nats block within each delta " \
                                          "computaionl chunk!!!"

    dA_cumsum = chunk_local_cumsum(g, chunk_size=64,
                           cu_seqlens=cu_seqlens,
                           )
    
    if decay_for_non_mamba_blocks:
        dA_cumsum = chunk_cumsum_non_gated_chunks(
            g_cumsum=dA_cumsum,
            chunk_size=chunk_size,
            nats_block_size=nats_block_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            chunk_indices_op_nats=chunk_indices_mamba_nats,
            reversed=False,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            offset_op=offset_mamba,
        )
    h, final_state = chunk_mamaba_nats_fwd_h(
        k=B,
        v=X,
        dA=dA_cumsum,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        nats_block_mamba_offsets=nats_block_mamba_offsets,
        chunk_indices_mamba_nats=chunk_indices_mamba_nats,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_mamba=offset_mamba,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
    )

    o = chunk_mamba_fwd_nats_o(
        q=C,
        k=B,
        v=X,
        h=h,
        g=dA_cumsum,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_op_nats=chunk_indices_mamba_nats,
        nats_block_op_offsets=nats_block_mamba_offsets,
        starting_h_idx=starting_h_idx_mamba,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        offset_op=offset_mamba,
        compute_incomplete_block_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        decay_for_non_mamba_blocks=decay_for_non_mamba_blocks
    )
    return o, dA_cumsum, final_state
    

def chunk_mamba2_bwd(
        X: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor,
        dA_cumsum: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor, 
        n_nats_blocks: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        chunk_indices_mamba_nats: torch.Tensor,
        nats_block_mamba_offsets: torch.Tensor,
        starting_h_idx_mamba: torch.Tensor,
        dt_softplus: bool = True,
        decay_for_non_mamba_blocks: bool= True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        nats_block_size: int = 64,
        offset_mamba: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        compute_dnats_for_invalid_blocks: bool = True,
        incomplete_block_start_with_ht: bool = True,
        chunk_size: int = 64,
        dt_limit=(0.0, float("inf")),
):
    h, final_st_ate = chunk_mamaba_nats_fwd_h(
        k=B,
        v=X,
        dA=dA_cumsum,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        nats_block_mamba_offsets=nats_block_mamba_offsets,
        chunk_indices_mamba_nats=chunk_indices_mamba_nats,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_mamba=offset_mamba,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
    )

    dX, qdo = chunk_bwd_dv_qdo_nats_local(q=C,
                                          k=B,
                                          do=do,
                                           v=X,
                                           nats_block_types=nats_block_types,
                                           nats_block_indices=nats_block_indices,
                                           n_nats_blocks=n_nats_blocks,
                                           chunk_indices_op_nats=chunk_indices_mamba_nats,
                                           g=dA_cumsum,
                                           scale=scale,
                                           cu_seqlens=cu_seqlens,
                                           cu_seqlens_nats=cu_seqlens_nats,
                                           chunk_size=chunk_size,
                                           nats_block_size=nats_block_size,
                                           offset_op=offset_mamba,
                                           compute_incomplete_block_scores=compute_incomplete_chunk_scores,
                                           pre_compute_qdo=True,
                                           incomplete_block_start_with_ht=incomplete_block_start_with_ht
                                           )
    
    dh, dh0, dX = chunk_mamba_nats_bwd_dhu(q=C,
                                           k=B,
                                           g=dA_cumsum,
                                           h0=initial_state,
                                           do=do,
                                           dht=dht,
                                           dv=dX,
                                           qdo=qdo,
                                           nats_block_types=nats_block_types,
                                           nats_block_indices=nats_block_indices,
                                           n_nats_blocks=n_nats_blocks,
                                           chunk_indices_mamba_nats=chunk_indices_mamba_nats,
                                           nats_block_mamba_offsets=nats_block_mamba_offsets,
                                           scale=scale,
                                           cu_seqlens=cu_seqlens,
                                           cu_seqlens_nats=cu_seqlens_nats,
                                           chunk_size=chunk_size,
                                           nats_block_size=nats_block_size,
                                           OFFSET_MAMBA=offset_mamba,
                                           compute_incomplete_chunk_scores=compute_incomplete_chunk_scores
                                           )

    dC, dB, ddA, d_nats = chunk_bwd_nats_dqkwg(q=C,
                                             k=B,
                                             v=X,
                                             do=do,
                                             h=h,
                                             dh=dh,
                                             nats_block_types=nats_block_types,
                                             nats_block_indices=nats_block_indices,
                                             nats_block_op_offsets=nats_block_mamba_offsets,
                                             n_nats_blocks=n_nats_blocks,
                                             starting_h_idx=starting_h_idx_mamba,
                                             chunk_indices_op_nats=chunk_indices_mamba_nats,
                                             g=dA_cumsum,
                                             cu_seqlens=cu_seqlens,
                                             cu_seqlens_nats=cu_seqlens_nats,
                                             chunk_size=chunk_size,
                                             scale=scale,
                                             nats_block_size=nats_block_size,
                                             offset_op=offset_mamba,
                                             compute_incomplete_block_scores=compute_incomplete_chunk_scores,
                                             compute_dnats_for_invalid_blocks=compute_dnats_for_invalid_blocks
                                             )
    if decay_for_non_mamba_blocks:
        ddA = chunk_cumsum_non_gated_chunks(
            g_cumsum=ddA,
            chunk_size=chunk_size,
            nats_block_size=nats_block_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            chunk_indices_op_nats=chunk_indices_mamba_nats,
            reversed=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            offset_op=offset_mamba,
        )
    ddA = chunk_local_cumsum(ddA, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    

    return dC, dB, dX, ddA, dh0, d_nats

    

def test_compute_h(dtype=torch.bfloat16):
    torch.manual_seed(0)
    from nats.utils import check_fp16_dtype
    import triton
    from torch.nn import functional as F
    from einops import rearrange
    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16
    dtype = torch.float16

    B = 2
    H = 4
    HNatS = 4
    T = 512
    N_TYPES = 3
    GNAtS = H // HNatS
    offset_mamba = 1
    K = 128
    V = 256
    NATS_Chunk = 64
    chunk_size = 64
    T_NAtS = triton.cdiv(T, NATS_Chunk)
    device = torch.device('cuda')
    q = torch.randn(B, T, H, K, dtype=dtype, device=torch.device('cuda'))
    k = torch.randn(B, T, H, K, dtype=dtype, device='cuda')
    v = torch.randn(B, T, H, V, dtype=dtype, device=torch.device('cuda'))
    #g0 = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device) * 20)
    A = torch.empty(H, dtype=torch.float32, device=torch.device('cuda')).uniform_(0, 16)
    A = torch.log(A)
    A = -torch.exp(A.float()) / 100
    dt = torch.randn(B, T, H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, device=device, dtype=dtype)

    logits = torch.randn(B, T_NAtS, HNatS, N_TYPES, device=torch.device('cuda'), dtype=dtype)
    nats_block_types = torch.nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
    # attN_TYPES[...,0] =1.
    # we ask all the models to have the last idx as valid
    nats_block_types[:, -1, :] = 1.
    nats_block_indices = torch.where(nats_block_types == 1.,
                                     torch.arange(T_NAtS, device=nats_block_types.device).view(1, -1, 1, 1), T_NAtS)
    nats_block_indices = nats_block_indices.sort(1)[0]
    n_nats_blocks = torch.sum(nats_block_types.long(), dim=1)
    decay_for_non_mamba_blocks = True
    import copy
    from mamba_ssm.ops.triton.ssd_combined import _chunk_cumsum_fwd, _chunk_state_fwd, _state_passing_fwd, _bmm_chunk_fwd, _chunk_scan_fwd
    output_final_state = True
    cu_seqlens = None
    mask_ = nats_block_types[..., offset_mamba]
    mask_ = mask_.repeat_interleave(NATS_Chunk, 1)
    mask_ = mask_[:, :, :, None].expand(B, NATS_Chunk * T_NAtS, HNatS, GNAtS).reshape(B, NATS_Chunk * T_NAtS,
                                                                                      HNatS * GNAtS)
    mask_ = mask_[:, :T, :]
    C = copy.deepcopy(q)
    B = copy.deepcopy(k) * mask_[..., None]
    x = copy.deepcopy(v) * mask_[..., None]
    # k1 = copy.deepcopy(k)
    # v1 = copy.deepcopy(v)
    # beta1 = copy.deepcopy(beta)
    import math
    scale = 1 

    #"""
    # delta net fwd


    dA_cumsum1, dt1 = _chunk_cumsum_fwd(copy.deepcopy(dt), copy.deepcopy(A), chunk_size, 
                                      dt_bias=copy.deepcopy(dt_bias), 
                                      dt_softplus=True, dt_limit=(0.0, float("inf")))
    dA_cumsum1 = dA_cumsum1*nats_block_types[..., offset_mamba].transpose(1,2)[..., None]

    states = _chunk_state_fwd(B, x, dt1, dA_cumsum1, states_in_fp32=True)

    states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum1[:, :, :, -1],
                                            chunk_size=chunk_size, out_dtype=C.dtype)
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n= B.shape[-1]) for t in [states, final_states]]
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=None, output_dtype=torch.float32)
    o1, out_x = _chunk_scan_fwd(CB, x, dt1, dA_cumsum1, C, states, D=None, z=None, seq_idx=None)
    
    #states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]


    # the followings are nats related !!!
    nats_block_size = NATS_Chunk
    offset_mamba = offset_mamba
    cu_seqlens_nats = None

    chunk_indices_mamba_nats = prepare_nats_block_indices(n_nats_blocks[..., offset_mamba],
                                                          nats_block_size,
                                                          chunk_size, )
    compute_incomplete_chunk_scores = True
    nats_block_mamba_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                          nats_block_types,
                                                          nats_block_size,
                                                          64, offset_mamba)

    #"""
    dA_cumsum2, dt2 = chunk_nats_cumsum_fwd(copy.deepcopy(dt), 
                                             copy.deepcopy(A), chunk_size=64,
                            dt_bias=dt_bias, dt_softplus=True, dt_limit=(0.0, float("inf"))
                           )
    """
    dA_cumsum2 = chunk_cumsum_non_gated_chunks(
            g_cumsum=dA_cumsum2,
            chunk_size=chunk_size,
            nats_block_size=nats_block_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            chunk_indices_op_nats=chunk_indices_mamba_nats,
            reversed=False,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            offset_op=offset_mamba,
        )
    """
    k =k*dt2

    h2, final_states2 = chunk_mamaba_nats_fwd_h(
        k=k,
        v=v,
        dA=dA_cumsum2,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        nats_block_mamba_offsets=nats_block_mamba_offsets,
        chunk_indices_mamba_nats=chunk_indices_mamba_nats,
        initial_state=None,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_mamba=offset_mamba,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=True,
        decay_for_non_mamba_blocks=True,
    )

    starting_h_idx_mamba = compute_starting_idx_for_chunks(
        nats_block_indices=nats_block_indices,
        T=T,
        BT=chunk_size,
        NAtS_Block_Size=nats_block_size,
        offset_op=offset_mamba
    )

    o2 = chunk_mamba_fwd_nats_o(
        q=q,
        k=k,
        v=v,
        h=h2,
        g=dA_cumsum2,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_op_nats=chunk_indices_mamba_nats,
        nats_block_op_offsets=nats_block_mamba_offsets,
        starting_h_idx=starting_h_idx_mamba,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        offset_op=offset_mamba,
        compute_incomplete_block_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=True,
        decay_for_non_mamba_blocks=False
    )


    import pdb
    pdb.set_trace()

    for b in range(len(o2)):
        for h0 in range(H):
            h_nats = h0 // GNAtS
            for i in range(triton.cdiv(T, chunk_size)):
                nats_block_type = nats_block_types[b, i, h0, offset_mamba]
                #o2_vanilla = o11[b, i * 64: i * 64 + 64, h0]
                #o2_vanilla = o1[b, i * 64: i * 64 + 64, h0]
                o1_nats = o2[b, i * 64: i * 64 + 64, h0]
                if nats_block_type > 0:
                    o2_vanilla = o1[b, i * 64: i * 64 + 64, h0]

                    diff = o1_nats - o2_vanilla
                    print(True)
                    print(f'diff with valid v_new: {diff.abs().max()} at b={b}, h={h0}, i={i}')
                else:
                    o2_vanilla = o11[b, i * 64: i * 64 + 64, h0]

                    xq = q[b, i * 64: i * 64 + 64, h0]
                    xk = k[b, i * 64: i * 64 + 64, h0]
                    xv = v[b, i * 64: i * 64 + 64, h0]
                    xg = dA_cumsum2[b, i * 64: i * 64 + 64, ]
                    dt3 = dt2[b, i * 64: i * 64 + 64, ]
                    b_A = xq @ xk.T

                    msk = torch.arange(0, 64).cuda()[:, None] >= torch.arange(0, 64).cuda()[None, :]
                    b_A = b_A * torch.exp(xg[:, None] - xg[None, :])  
                    #xv = xv * dt3[None, :]
                    b_A = torch.where(msk, b_A, 0)
                    if decay_for_non_mamba_blocks and i>0 and nats_block_types[b, i-1, h0, offset_mamba]:
                        h_tmp = h1[b, i, h0]
                    else:
                        h_tmp = h1[b, i, h0]
                    g = dA_cumsum2[b, i * 64: i * 64 + 64, ]
                    dt0 = dt2[b, i * 64: i * 64 + 64, ]
                    

                    diff = o1_nats - (o2_vanilla + b_A.to(h1) @ xv)
                    print(False)
                    print(f'diff with valid v_new: {diff.abs().max()} at b={b}, h={h0}, i={i}')

    import pdb
    pdb.set_trace()


def test_bwd():
    torch.manual_seed(0)
    from nats.utils import check_fp16_dtype
    import triton
    from torch.nn import functional as F
    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16
    dtype = torch.float16

    B = 2
    H = 4
    HNatS = 4
    T = 512
    N_TYPES = 3
    GNAtS = H // HNatS
    delta_offset = 1
    K = 128
    V = 256
    NATS_Chunk = 64
    chunk_size = 64
    T_NAtS = triton.cdiv(T, NATS_Chunk)
    device = torch.device('cuda')
    q = torch.randn(B, T, H, K, dtype=dtype, device=torch.device('cuda'))
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device='cuda'), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=dtype, device=torch.device('cuda'))
    do = torch.randn(B, T, H, V, dtype=dtype, device=torch.device('cuda'))
    #g0 = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device) * 20)
    A = torch.empty(H, dtype=torch.float32, device=torch.device('cuda')).uniform_(0, 16)
    A = torch.log(A)
    A = -torch.exp(A.float()) / 100
    dt0 = torch.randn(B, T, H, dtype=torch.float32, device=device)
    dt_bias = torch.randn(H, device=device, dtype=dtype)

    dht = torch.randn(B, H, K, V, dtype=torch.float32, device=torch.device('cuda'))
    h0 = torch.zeros(B, H, K, V, dtype=torch.float32, device=torch.device('cuda'))

    logits = torch.randn(B, T_NAtS, HNatS, N_TYPES, device=torch.device('cuda'), dtype=dtype)
    nats_block_types = torch.nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
    nats_block_types[:, -1, ] = 1.
    # attN_TYPES[...,0] =1.
    nats_block_indices = torch.where(nats_block_types == 1.,
                                     torch.arange(T_NAtS, device=nats_block_types.device).view(1, -1, 1, 1), T_NAtS)
    nats_block_indices = nats_block_indices.sort(1)[0]
    n_nats_blocks = torch.sum(nats_block_types.long(), dim=1)

    import copy
    from mamba_ssm.ops.triton.ssd_combined import _chunk_cumsum_fwd, _chunk_state_fwd, _state_passing_fwd, _bmm_chunk_fwd, _chunk_scan_fwd
    from mamba_ssm.ops.triton.ssd_combined import (_chunk_scan_bwd_dstates,  _state_passing_bwd, _chunk_scan_chunk_state_bwd_dx, _chunk_state_bwd_db, 
                                                   _chunk_scan_bwd_dC, _bmm_chunk_bwd, _chunk_scan_bwd_ddAcs_stable, _chunk_cumsum_bwd, _chunk_scan_bwd_dcb)
    
    from einops import rearrange

    output_final_state = True
    cu_seqlens = None
    mask_ = nats_block_types[..., delta_offset]
    mask_ = mask_.repeat_interleave(NATS_Chunk, 1)
    mask_ = mask_[:, :, :, None].expand(B, NATS_Chunk * T_NAtS, HNatS, GNAtS).reshape(B, NATS_Chunk * T_NAtS,
                                                                                      HNatS * GNAtS)
    mask_ = mask_[:, :T, :]
    C = copy.deepcopy(q)
    B = copy.deepcopy(k) * mask_[..., None]
    x = copy.deepcopy(v) * mask_[..., None]
    #"""
    import math
    scale = 1 
    seq_idx = None
    dstate = B.shape[-1]
    dout = copy.deepcopy(do)
    dt_softplus= True
    dt_limit = (0.0, float("inf"))

    dA_cumsum0, dt = _chunk_cumsum_fwd(copy.deepcopy(dt0), copy.deepcopy(A), chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)
    dA_cumsum = dA_cumsum0*nats_block_types[..., delta_offset].transpose(1,2)[..., None]
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                   initial_states=None,
                                   seq_idx=None, chunk_size=chunk_size)
    states = rearrange(states, "... (p n) -> ... p n", n=B.shape[-1])

    dz = None
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum0, dout, seq_idx=None, dtype=states.dtype)
    qdo_0 = copy.deepcopy(dstates)
    # dstates has length nchunks, containing the gradient to initial states at index 0 and
    # gradient to the states of chunk (nchunks - 2) at index (nchunks - 1)
    # Do computation in fp32 but convert dstates and states to fp16/bf16 since dstates and states
    # will be used in matmul in the next kernels.
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=None,
        seq_idx=seq_idx,
        has_initial_states=False,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    # dstates has length nchunks, containing the gradient to states of chunk 0 at index 0 and
    # gradient to the final states at index (nchunks - 1)
    # states has length nchunks, containing the initial states at index 0 and the state for chunk (nchunks - 2) at index (nchunks - 1)
    # The final states is not stored.
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    D = None
    ngroups = B.shape[-2]
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum0, B, CB, dout, dstates, D=None, seq_idx=seq_idx, dx=None)
    # dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, ngroups=ngroups)
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=B.shape[-2])
    # dC = _chunk_scan_bwd_dC(states[:, :-1].to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups)
    # Computing ddA with the dcb kernel is much slower, so we're not using it for now
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    # dCB, ddA_tmp = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, CB=CB, ngroups=ngroups)
    dCB = dCB.to(CB.dtype)
    dB_given = torch.empty_like(B)
    dC_given = torch.empty_like(C)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)
    # If we have z, then dout_x is recomputed in fp32 so dD = (dout_x * x).sum() is more accurate
    # than dD_from_x = (dout_x * x).sum() where dout_x is in fp16/bf16
    dD = dD_from_x
    # Formula for ddA_cumsum, assuming out is the output of the forward pass before adding x * D.
    # ddA_cumsum = torch.einsum("bclhp,bclhp->bhcl", out.float(), dout.float()) - ddt * dt
    # However, this is numerically unstable: when we do the reverse cumsum on ddA_cumsum, there might
    # be a lot of underflow.

    # This is already done as part of bwd_dC kernel
    # ddA_cumsum_prev = _chunk_scan_bwd_ddAcs_prev(states[:, :-1], C, dout, dA_cumsum, seq_idx=seq_idx)
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    # This is already done as part of bwd_dB kernel
    # ddA_next = _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates, seq_idx=seq_idx)
    # We don't need to pass in seq_idx because CB also zeros out entries where seq_idx[i] != seq_idx[j]
    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev

    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt, dt0.clone(), A, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit, ddt=None)
    #"""
    scale = 1

    nats_block_size = NATS_Chunk
    offset_mamba = delta_offset
    cu_seqlens_nats = None

    chunk_indices_mamba_nats = prepare_nats_block_indices(n_nats_blocks[..., delta_offset],
                                                          nats_block_size,
                                                          chunk_size, )
    compute_incomplete_chunk_scores = True
    nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks, nats_block_types, nats_block_size,
                                                          chunk_size, offset_mamba)
    dA_cumsum2, dt2 = chunk_nats_cumsum_fwd(copy.deepcopy(dt0), 
                                             copy.deepcopy(A), chunk_size=64,
                            dt_bias=dt_bias, dt_softplus=True, dt_limit=(0.0, float("inf"))
                           )
    nats_block_mamba_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                          nats_block_types,
                                                          nats_block_size,
                                                          64, offset_mamba)
    decay_for_non_mamba_blocks = False
    initial_state = None
    incomplete_block_start_with_ht = True
    v *= dt2.unsqueeze(-1)
    h2, final_state2 = chunk_mamaba_nats_fwd_h(
        k=k,
        v=v,
        dA=dA_cumsum2,
        #dt=dt2,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        nats_block_mamba_offsets=nats_block_mamba_offsets,
        chunk_indices_mamba_nats=chunk_indices_mamba_nats,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_mamba=offset_mamba,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
    )
    starting_h_idx_mamba = compute_starting_idx_for_chunks(
        nats_block_indices=nats_block_indices,
        T=T,
        BT=chunk_size,
        NAtS_Block_Size=nats_block_size,
        offset_op=offset_mamba
    )
    cu_seqlens = None
    cu_seqlens_nats = None
    dv, qdo = chunk_bwd_dv_qdo_nats_local(q=q,
                                          k=k,
                                          do=do, 
                                          v=v,
                                           nats_block_types=nats_block_types,
                                           nats_block_indices=nats_block_indices,
                                           n_nats_blocks=n_nats_blocks,
                                           chunk_indices_op_nats=chunk_indices_mamba_nats,
                                           g=dA_cumsum2,
                                           scale=scale,
                                           cu_seqlens=cu_seqlens,
                                           cu_seqlens_nats=cu_seqlens_nats,
                                           chunk_size=chunk_size,
                                           nats_block_size=nats_block_size,
                                           offset_op=offset_mamba,
                                           compute_incomplete_block_scores=compute_incomplete_chunk_scores,
                                           pre_compute_qdo=True,
                                           incomplete_block_start_with_ht=incomplete_block_start_with_ht
                                           )
    dh, dh0, dv = chunk_mamba_nats_bwd_dhu(q=q,
                                           k=k,
                                           g=dA_cumsum2,
                                           h0=initial_state,
                                           do=do,
                                           dht=None,
                                           dv=dv,
                                           qdo=qdo,
                                           nats_block_types=nats_block_types,
                                           nats_block_indices=nats_block_indices,
                                           n_nats_blocks=n_nats_blocks,
                                           chunk_indices_mamba_nats=chunk_indices_mamba_nats,
                                           nats_block_mamba_offsets=nats_block_mamba_offsets,
                                           scale=scale,
                                           cu_seqlens=cu_seqlens,
                                           cu_seqlens_nats=cu_seqlens_nats,
                                           chunk_size=chunk_size,
                                           nats_block_size=nats_block_size,
                                           OFFSET_MAMBA=offset_mamba,
                                           compute_incomplete_chunk_scores=compute_incomplete_chunk_scores
                                           )

    #dv = dv * dt2[..., None]

    dq, dk, ddA2, d_nats = chunk_bwd_nats_dqkwg(q=q,
                                             k=k,
                                             v=v,
                                             do=do,
                                             h=h2,
                                             dh=dh,
                                             nats_block_types=nats_block_types,
                                             nats_block_indices=nats_block_indices,
                                             nats_block_op_offsets=nats_block_mamba_offsets,
                                             n_nats_blocks=n_nats_blocks,
                                             starting_h_idx=starting_h_idx_mamba,
                                             chunk_indices_op_nats=chunk_indices_mamba_nats,
                                             g=dA_cumsum2,
                                             #dt=dt2,
                                             cu_seqlens=cu_seqlens,
                                             cu_seqlens_nats=cu_seqlens_nats,
                                             chunk_size=chunk_size,
                                             scale=scale,
                                             nats_block_size=nats_block_size,
                                             offset_op=offset_mamba,
                                             compute_incomplete_block_scores=compute_incomplete_chunk_scores,
                                             compute_dnats_for_invalid_blocks=False
                                             )
    i_start = 0
    g2 = dA_cumsum2
    offset_delta = offset_mamba
    for b in range(len(dq)):
        for h0 in range(H):
            h_nats = h0 % GNAtS
            print(f"b: {b}, h {h0}")
            for i in range(T_NAtS):
                h_cur = h2[i_start, h_nats]
                dh_cur = dh[i_start, h_nats].T
                if nats_block_types[b, i, h0, offset_delta]:
                    print(f' delta ops')
                else:
                    print(f'non delta ops')
                v_current = v[b, i * 64:(i + 1) * 64, h0]

                b_do = do[b, i * 64:(i + 1) * 64, h0]

                b_ds = b_do @ v_current.T

                ot = torch.arange(0, 64).cuda()
                b_g = g2[b, i * 64:(i + 1) * 64, h0]
                b_ds = torch.where(ot[:, None] >= ot[None, :], b_ds * torch.exp(b_g[:, None] - b_g[None, :]),
                                   0) * scale

                q_cur = q[b, i * 64:(i + 1) * 64, h0]
                k_cur = k[b, i * 64:(i + 1) * 64, h0]
                b_dq = b_do @ h_cur.T
                if nats_block_types[b, i, h0, offset_delta]:
                    b_dk = v_current @ dh_cur
                    b_dk = b_dk * torch.exp(-b_g + b_g[-1])[:, None]
                else:
                    b_dk = 0
                b_dq = b_dq * torch.exp(b_g)[:, None] * scale

                b_dnats = (k_cur * (v_current @ dh_cur)).sum() / 64
                b_ds = b_ds.to(q_cur)
                if  nats_block_types[b, i, h0, offset_delta]:
                    import pdb
                    pdb.set_trace()
                b_dq += b_ds @ k_cur
                b_dk += b_ds.T @ q_cur

                diff_dq = (dq[b, i * 64:(i + 1) * 64, h0] - b_dq).abs()
                diff_dk = (dk[b, i * 64:(i + 1) * 64, h0] - b_dk).abs()
                diff_dnats = (d_nats[b, i, h0] - b_dnats).abs()
                print(f"diff qdq:{diff_dq.max()}")
                print(f"diff dk:{diff_dk.max()}")
                # print(f"diff dnats: {diff_dnats}")
                if nats_block_types[b, i, h0, offset_delta]:
                    i_start += 1

                """
                dq_nats = dq[b, i * 64:i * 64 + 64, h0]
                dk_nats = dk[b, i * 64:i * 64 + 64, h0]

                dq_raw = dq1[b, i * 64:i * 64 + 64, h0]
                dk_raw = dk1[b, i * 64:i * 64 + 64, h0]

                if nats_block_types[b, i, h0, 1] != 0.:
                    print(f' delta ops')
                    print(f"dq diff{(dq_nats - dq_raw).abs().max()}")
                    print(f"dk diff{(dk_nats - dk_raw).abs().max()}")


                else:
                    print(f'non delta ops')
                    b_ds = do[b, i * 64:(i + 1) * 64, h0] @ v_new[b, i * 64:(i + 1) * 64, h0].T
                    ot = torch.arange(0, 64).cuda()
                    b_g = g2[b, i * 64:(i + 1) * 64, h0]
                    b_ds = torch.where(ot[:, None] >= ot[None, :], b_ds * torch.exp(b_g[:, None] - b_g[None, :]),
                                       0) * scale

                    q_cur = q[b, i * 64:(i + 1) * 64, h0]
                    k_cur = k[b, i * 64:(i + 1) * 64, h0]
                    b_ds = b_ds.to(q_cur)
                    print(f"dq diff: {(dq_raw + b_ds @ k_cur - dq_nats).abs().max()}")
                    print(f"dk diff: {(b_ds.T @ q_cur - dk_nats).abs().max()}")
                """

    import pdb
    pdb.set_trace()
    if decay_for_non_mamba_blocks:
        dd2 = chunk_cumsum_non_gated_chunks(
            g_cumsum=ddA2,
            chunk_size=chunk_size,
            nats_block_size=nats_block_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            chunk_indices_op_nats=chunk_indices_mamba_nats,
            reversed=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            offset_op=offset_mamba,
        )
    
    ddt_given2, ddA2, ddt_bias2 = chunk_nats_cumsum_bwd(
        ddA2, ddt2, dt=dt0.clone(), chunk_size=chunk_size,
        A=A, dt_bias=dt_bias, 
        dt_softplus=dt_softplus, dt_limit=dt_limit, ddt=None
        )
    
    import pdb
    pdb.set_trace()

    # torch.save({'dv': dv, 'qdo': qdo}, 'qdo.pth')
    # """
    # """
    i_start = 0

    for b in range(len(dv)):
        for h in range(H):
            h_nats = h % GNAtS
            nats_chunk_idx = nats_block_indices[b, :, h, offset_mamba].view(-1, triton.cdiv(64,
                                                                                            nats_block_size)) * nats_block_size
            for i in range(triton.cdiv(T, 64)):
                xq = q[b, i * 64: i * 64 + 64, h]
                xdo = do[b, i * 64: i * 64 + 64, h]
                xg = dA_cumsum2[b, i * 64: i * 64 + 64, h]
                qdo_vanilla = (xq.T.float() * scale * torch.exp(xg[None, :])).to(dtype) @ xdo
                diff = qdo_0[b,i,h].T-qdo_vanilla
                print(f'***' * 50)

                print(b)
                print(i)
                print(h)
                print('mamba!')
                print(diff.abs().max())
                print(f'***' * 50)

                if i == 0:
                    if nats_chunk_idx[i, 0] != 0:
                        idx_start = 0
                    else:
                        idx_start = 64
                else:
                    idx_start = nats_chunk_idx[i - 1, 0] + 64
                if idx_start >= T:
                    continue
                # if nats_chunk_idx[i+1, 0]>=T and nats_block_types[b,-1,h, offset_mamba] == 0:
                if False:
                    # in this case we need another opeartino
                    idx_end = nats_chunk_idx[i, 0]
                    q_indices = torch.arange(idx_start, idx_end).cuda()
                    qdo_nats = qdo[i_start]
                    xq = q[b, q_indices, h]
                    xg = g2[b, q_indices, h]
                    xdo = do[b, q_indices, h]
                    qdo_vanilla = (xq.T.float() * scale * torch.exp(xg[None, :])).to(dtype) @ xdo
                    diff_qdo = qdo_vanilla - qdo_nats
                    print(f'!!!' * 50)
                    print(idx_start)
                    print(idx_end)
                    print(diff_qdo.abs().max())

                    i_start += 1
                    idx_start = nats_chunk_idx[i, 0] + 64
                    idx_end = nats_chunk_idx[i + 1, 0]
                    q_indices = torch.arange(idx_start, idx_end).cuda()
                    qdo_nats = qdo[i_start]
                    xq = q[b, q_indices, h]
                    xg = g2[b, q_indices, h]
                    xdo = do[b, q_indices, h]
                    qdo_vanilla = (xq.T.float() * scale * torch.exp(xg[None, :])).to(dtype) @ xdo
                    diff_qdo = qdo_vanilla - qdo_nats
                    print(f'????' * 50)
                    print(idx_start)
                    print(idx_end)
                    print(diff_qdo.abs().max())
                else:
                    idx_end = min(nats_chunk_idx[i, 0] + 64, T) - 64
                    qdo_nats = qdo[i_start]

                    if idx_end > idx_start:
                        # idx_end = min(nats_chunk_idx[i,0] + 64, T)
                        k_indices = (nats_chunk_idx[i].unsqueeze(1) + torch.arange(nats_block_size).cuda().unsqueeze(
                            0)).flatten()
                        q_indices = torch.arange(idx_start, idx_end).cuda()

                        xq = q[b, q_indices, h]
                        xk = k[b, torch.where(k_indices >= T, 0, k_indices), h]
                        xdo = do[b, q_indices, h]
                        msk = k_indices[:, None] <= q_indices[None, :]
                        # dv_nats = dv[i_start*64:i_start*64+64, h_nats]
                        # xgq = g0[b, q_indices, h]
                        # xg = g2[i_start*64:i_start*64+64, h_nats]
                        xg = dA_cumsum2[b, q_indices, h]
                        # dv_vanilla = (torch.where(msk, xk@xq.T, 0).float() * scale * torch.exp(xgq[None, :] - xg[:, None])).to(dtype) @ xdo
                        qdo_vanilla = (xq.T.float() * scale * torch.exp(xg[None, :])).to(dtype) @ xdo

                    else:
                        qdo_vanilla = 0

                    # diff_dv = dv_nats-dv_vanilla
                    diff_qdo = qdo_vanilla - qdo_nats
                    print(f'***' * 50)

                    print(idx_start)
                    print(idx_end)
                    print(diff_qdo.abs().max())
                    # import pdb
                    # pdb.set_trace()



                i_start += 1
    import pdb
    pdb.set_trace()

    # """
    # """
    # torch.save({'dv':dv, 'qdo':qdo}, 'qdo.pth')

    # """

    """

    #dvqdo = torch.load('qdo.pth')

    #dv = dvqdo['dv']
    #qdo = dvqdo['qdo']

    # import pdb
    # pdb.set_trace()
    """
    # """
    # """
    i_start = 0
    """
    torch.save({
        'dh':dh,
        'dv01': dv01
    },
    'dh.pth'
    )
    """
    """
    """
    dh1 = dstates.transpose(3,4)
    for b in range(len(dv)):
        for ih in range(H):
            h_nats = ih % GNAtS
            nats_chunk_idx = nats_block_indices[b, :, ih, offset_mamba]
            for i in range(triton.cdiv(T, 64) - 1):
                if nats_chunk_idx[i + 1] == triton.cdiv(T, 64) and nats_chunk_idx[i] < T_NAtS - 1:
                    dhnats = dh[i_start, 0]
                    dh_raw = dh1[b, nats_chunk_idx[i], ih]
                    diff = dhnats - dh_raw

                    print(f'***' * 50)

                    print(b)
                    print(ih)
                    print(nats_chunk_idx[i])
                    print(diff.abs().max())
                    i_start += 1

                    dhnats = dh[i_start, 0]
                    dh_raw = dh1[b, nats_chunk_idx[i] + 1, ih]
                    diff = dhnats - dh_raw

                    print(f'***' * 50)

                    print(b)
                    print(ih)
                    print(nats_chunk_idx[i] + 1)
                    print(diff.abs().max())
                    i_start += 1

                if nats_chunk_idx[i] == triton.cdiv(T, 64):
                    continue
                dhnats = dh[i_start, 0]
                dh_raw = dh1[b, nats_chunk_idx[i], ih]
                diff = dhnats - dh_raw
                print(f'***' * 50)

                print(b)
                print(ih)
                print(nats_chunk_idx[i])
                print(diff.abs().max())
                i_start += 1


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    #test_compute_h()
    test_bwd()


    


    

    
    