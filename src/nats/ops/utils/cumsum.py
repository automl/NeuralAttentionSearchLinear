# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from fla.ops.utils.cumsum import chunk_local_cumsum_scalar
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import check_shared_mem, input_guard, is_nvidia_hopper

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        # triton.Config({'BV': BV, 'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        # for BV in [128, 64,]
        # for BK in [128, 64,]
        for num_warps in  NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'USE_G'],
)
@triton.heuristics({
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT']),
})
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def chunk_cumsum_non_gated_chunks_kernel(
        g_cumsum,
        nats_block_types,
        nats_block_indices,
        cu_seqlens,
        cu_seqlens_nats,
        chunk_indices_op_nats,
        T,
        TNAtS,
        H: tl.constexpr,
        HNAtS: tl.constexpr,
        GNAtS: tl.constexpr,
        BT: tl.constexpr,
        NAtS_BLOCK_SIZE: tl.constexpr,
        OFFSET_OP: tl.constexpr,
        REVERSED: tl.constexpr,
        N_TYPES: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
):
    # and goes towards the end of last nats block
    i_t_, i_gnats = tl.program_id(1), tl.program_id(2)

    off_bh_nats = tl.load(chunk_indices_op_nats + i_t_ * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_op_nats + i_t_ * 2 + 1).to(tl.int32)
    i_b = off_bh_nats // HNAtS
    i_hnats = off_bh_nats % HNAtS

    i_h = i_hnats * GNAtS + i_gnats

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

    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP
    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    stride_block_types_t = N_TYPES * HNAtS
    stride_g_t = H

    g_cumsum += bos * H + i_h
    load_idx_chunk = i_t
    i_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
    i_nats_block_last = tl.load(
        nats_block_indices + (load_idx_chunk - 1) * stride_block_types_t,
        mask=load_idx_chunk > 1, other=-1
    ).to(tl.int32)  # if the current block is the first one, we do not need to do anything, otherwie,

    # we only need the cases where i_nats_block_next - i_nats_block > 2, otherwise, if there is only one gated
    # delta block inbetween, there is no need to do this cumulative computing

    if N_CHUNK_PER_NAtS_BLOCK > 1:
        # TODO this is only a draft implementation, we need to check the correctness of this function!
        # In this case, we also need to load the first chunk values
        #i_nats_block_last = i_nats_block_last + 1
        if REVERSED:
            load_msk_first = tl.arange(0, N_CHUNK_PER_NAtS_BLOCK) < (N_CHUNK_PER_NAtS_BLOCK - 1) & (i_nats_block_last > 1)[None, :]
            bg_first_value = tl.load(
                g_cumsum + stride_g_t * (BT * tl.arange(0, N_CHUNK_PER_NAtS_BLOCK) + i_nats_block * NAtS_BLOCK_SIZE - NAtS_BLOCK_SIZE), mask=load_msk_first, other=0
            )
            pg_cumsum = tl.make_block_ptr(
                g_cumsum + stride_g_t * (i_nats_block * NAtS_BLOCK_SIZE - NAtS_BLOCK_SIZE), (T), (stride_g_t, ), (0, ), (NAtS_BLOCK_SIZE, ), (0, )
            )
            bg_first_values_cumsum = tl.cumsum(bg_first_value, 0)

            bg_first_value_sum = tl.sum(bg_first_values_cumsum)
            bg_first_values_cumsum_expanded = tl.reshape(
                tl.broadcast_to(bg_first_values_cumsum[:, None], (N_CHUNK_PER_NAtS_BLOCK, BT)), [])
            bg_cumsum = tl.load(pg_cumsum, boundary_check=(0,))
            bg_cumsum += bg_first_values_cumsum_expanded
            tl.store(bg_cumsum.to(pg_cumsum.dtype.element_ty), boundary_check=(0,))
            n_iters = i_nats_block - i_nats_block_last - 1
            for i in n_iters:
                pg_first_values = tl.make_block_ptr(
                    g_cumsum + (i_nats_block - 1 - i) * stride_g_t * NAtS_BLOCK_SIZE, (T, ),
                    (stride_g_t * BT), (0,), (N_CHUNK_PER_NAtS_BLOCK,),
                )
                bg_first_values = tl.load(pg_first_values)
                bg_first_values_cumsum = tl.cumsum(bg_first_values, 0, )
                bg_first_values_cumsum += bg_first_value_sum
                bg_first_value_sum += tl.sum(bg_first_values)

                pg_cumsum = tl.make_block_ptr(
                    g_cumsum  + (i_nats_block - 1 - i) * stride_g_t * NAtS_BLOCK_SIZE, (T, ), (stride_g_t,), (i * NAtS_BLOCK_SIZE,),
                    (NAtS_BLOCK_SIZE,), (0,)
                )
                bg_cumsum = tl.load(pg_cumsum, boundary_check=(0,))
                bg_first_values_cumsum_expanded = tl.reshape(
                    tl.broadcast_to(bg_first_values_cumsum[:, None], (N_CHUNK_PER_NAtS_BLOCK, BT)), [])
                bg_cumsum += bg_first_values_cumsum_expanded
                tl.store(pg_cumsum, bg_cumsum.to(pg_cumsum.dtype.element_ty), boundary_check=(0,))
        else:
            i_nats_block_last = tl.where(i_nats_block_last < 0, 0, i_nats_block_last)
            #n_iters = i_nats_block - i_nats_block_last

            g_cumsum += stride_g_t * i_nats_block_last * NAtS_BLOCK_SIZE
            load_msk_first = (tl.arange(0, N_CHUNK_PER_NAtS_BLOCK)) > 0 & \
                              (tl.arange(0, N_CHUNK_PER_NAtS_BLOCK) * BT + i_nats_block_last * NAtS_BLOCK_SIZE < T)
            bg_last_values = tl.load(
                g_cumsum + stride_g_t * (BT * tl.arange(0, N_CHUNK_PER_NAtS_BLOCK) + T-1), mask=load_msk_first, other=0
            )
            pg_cumsum = tl.make_block_ptr(
                g_cumsum, (T - i_nats_block_last * NAtS_BLOCK_SIZE), (stride_g_t, ), (0, ), (NAtS_BLOCK_SIZE, ), (0, )
            )
            bg_last_values_cumsum = tl.cumsum(bg_last_values, 0)
            bg_last_value_sum = tl.sum(bg_last_values)
            bg_last_values_cumsum_expanded = tl.reshape(tl.broadcast_to(bg_last_values_cumsum[:, None], (N_CHUNK_PER_NAtS_BLOCK, BT)), [])
            bg_cumsum = tl.load(pg_cumsum, boundary_check=(0,))
            bg_cumsum += bg_last_values_cumsum_expanded
            tl.store(bg_cumsum.to(pg_cumsum.dtype.element_ty), boundary_check=(0,))
            n_iters = i_nats_block - i_nats_block_last - 1
            for i in n_iters:
                pg_last_values = tl.make_block_ptr(
                    g_cumsum + (i + 1) * stride_g_t * NAtS_BLOCK_SIZE, (T - i_nats_block_last * NAtS_BLOCK_SIZE),
                    (stride_g_t * BT), (0, ), (N_CHUNK_PER_NAtS_BLOCK, ),
                )
                bg_last_values = tl.load(pg_last_values)
                bg_last_values_cumsum = tl.cumsum(bg_last_values, 0,)
                bg_last_values_cumsum += bg_last_value_sum
                bg_last_value_sum += tl.sum(bg_last_values)

                pg_cumsum = tl.make_block_ptr(
                    g_cumsum, (T - i_nats_block_last * NAtS_BLOCK_SIZE), (stride_g_t,), (i * NAtS_BLOCK_SIZE,), (NAtS_BLOCK_SIZE,), (0,)
                )
                bg_cumsum = tl.load(pg_cumsum, boundary_check=(0,))
                bg_last_values_cumsum_expanded = tl.reshape(
                    tl.broadcast_to(bg_last_values_cumsum[:, None], (N_CHUNK_PER_NAtS_BLOCK, BT)), [])
                bg_cumsum += bg_last_values_cumsum_expanded
                tl.store(pg_cumsum, bg_cumsum.to(pg_cumsum.dtype.element_ty), boundary_check=(0,))
    else:
        i_nats_block_last += 1  # we start with the first gated block and extract its last element
        n_iters = i_nats_block - i_nats_block_last - 1
        if REVERSED:
            bg_first_cumsum = tl.load(g_cumsum + (i_nats_block -1) * BT *stride_g_t , mask=i_nats_block > 1, other=0)
            for i in range(n_iters):
                bg_first = tl.load(g_cumsum, (i_nats_block_last + i) * BT)
                p_g = tl.make_block_ptr(
                    g_cumsum, (T,), (stride_g_t,), ((i_nats_block_last + i) * BT,), (BT,), (0,)
                )
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g += bg_first_cumsum
                tl.store(p_g, b_g.to(p_g.dtype.element_ty), boundary_check=(0,))
                bg_first_cumsum += bg_first
        else:
            last_idx = min(i_nats_block_last * BT + BT, T) - 1
            bg_last_cumsum = tl.load(g_cumsum + last_idx * stride_g_t, )
            for i in range(n_iters):
                last_idx = min((i_nats_block_last + i) * BT + BT, T) - 1
                bg_last = tl.load(g_cumsum + last_idx * stride_g_t, )
                p_g = tl.make_block_ptr(
                    g_cumsum, (T, ), (stride_g_t, ), ((i_nats_block_last + i) * BT, ), (BT, ), (0,)
                )
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g += bg_last_cumsum
                tl.store(p_g, b_g.to(p_g.dtype.element_ty), boundary_check=(0,))
                bg_last_cumsum += bg_last


def chunk_cumsum_non_gated_chunks(g_cumsum: torch.Tensor,
                                  chunk_size: int,
                                  nats_block_size: int,
                                  nats_block_types:torch.Tensor,
                                  nats_block_indices: torch.Tensor,
                                  chunk_indices_op_nats: torch.Tensor,
                                  reversed: bool = False,
                                  scale: float = None,
                                  cu_seqlens: Optional[torch.Tensor] = None,
                                  cu_seqlens_nats:Optional[torch.Tensor] = None,
                                  offset_op: int=0,
                                  head_first: bool = False,
                                  output_dtype: Optional[torch.dtype] = torch.float,
                                  ):
    """
    This function is used to compute g_cumsum for the non-gated blocks. THis is applied for the cases where
    the decay still happens even if the target block is not the gated delta net. Hence, this funciton first
    need to compute all the cumulative sum of the non-gated blocks,
    Args:
        g_cumsum:
        chunk_size:
        nats_block_types:
        nats_block_indices:
        chunk_indices_op_nats:
        reversed:
        scale:
        cu_seqlens:
        head_first:
        output_dtype:

    Returns:

    """
    B, T, H = g_cumsum.shape
    B, TNAtS, HNAtS, n_opts = nats_block_types.shape
    GNAtS = H // HNAtS

    grid = (len(chunk_indices_op_nats), GNAtS)
    chunk_cumsum_non_gated_chunks_kernel[grid](g_cumsum, nats_block_types, nats_block_indices, cu_seqlens, cu_seqlens_nats,
                                        chunk_indices_op_nats, T, TNAtS, H, HNAtS, GNAtS, BT=chunk_size,
                                        NAtS_BLOCK_SIZE=nats_block_size, REVERSED=reversed,
                                        OFFSET_OP=offset_op, N_TYPES=n_opts
                                        )
    return g_cumsum