import warnings

import torch
import triton
import triton.language as tl

from einops import reduce

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.op import exp2, log2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous

from nats.utils import check_fp16_dtype
from nats.ops.nats_util import prepare_nats_block_indices, compute_attn_n_iters_per_block

fp16_type = tl.bfloat16 if check_fp16_dtype() == 'bfloat16' else tl.float16


@triton.jit
def exp2(x):
    return tl.math.exp2(x)


@triton.jit
def log2(x):
    return tl.math.log2(x)


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


"""
@triton.jit
def compute_mask_lower(start_n: tl.constexpr, NAtS_block_size: tl.constexpr, N_NAtS_Chunk_PER_S: tl.constexpr, offs_m):
    offs_n_lower = (start_n + tl.arange(0, NAtS_block_size) * N_NAtS_Chunk_PER_S + NAtS_block_size)
    mask_lower = offs_n_lower[None, :] <= offs_m[:, None]
    return mask_lower
"""

@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BS']),
    'USE_SW': lambda args: args['SW_SIZE'] is not None,
    'COMPUTE_PARTIAL_CHUNKS': lambda args: args['USE_SW'] or args['COMPUTE_INCOMPLETE_CHUNK_SCORES']
})
@triton.jit
def parallel_nats_attn_fwd_kernel(q, k, v, o,  # Q, O are of shape [B, T, H, DQ]
                                  msk,
                                  g_cumsum, lse,
                                  nats_block_types, nats_block_indices,  # Both are of shape [B, TNAtSBlock, H]
                                  n_iters_per_block,
                                  cu_seqlens, chunk_indices, cu_seqlens_nats,
                                  scale,
                                  T: tl.constexpr,
                                  TNAtS: tl.constexpr,
                                  B: tl.constexpr,
                                  HQ: tl.constexpr,
                                  HKV: tl.constexpr,
                                  G: tl.constexpr,
                                  HNAtS: tl.constexpr,
                                  GNAtS: tl.constexpr,
                                  K: tl.constexpr,
                                  V: tl.constexpr,
                                  BT: tl.constexpr,
                                  BS: tl.constexpr,
                                  BK: tl.constexpr,
                                  BV: tl.constexpr,
                                  STRIDE_KB: tl.constexpr,
                                  STRIDE_VB: tl.constexpr,
                                  NAtS_BLOCK_SIZE: tl.constexpr,
                                  SW_SIZE:tl.constexpr,
                                  N_TYPES: tl.constexpr,
                                  OFFSET_ATTN: tl.constexpr,
                                  COMPUTE_INCOMPLETE_CHUNK_SCORES: tl.constexpr,
                                  STAGE: tl.constexpr,
                                  USE_G: tl.constexpr,
                                  IS_VARLEN: tl.constexpr,
                                  N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
                                  USE_SW: tl.constexpr,
                                  COMPUTE_PARTIAL_CHUNKS: tl.constexpr,
                                  STORE_MSK: tl.constexpr
                                  ):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    i_hnats = i_hq // GNAtS

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(
            tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        # In this case we might just assume that each operation contains one dim or
        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
        # TODO we need to check if this is correct!!!
        # b_n_iters = tl.load(n_iters_per_block + i_n * N_types + OFFSET_ATTN).to(tl.int32)

        n_iters_per_block = n_iters_per_block + i_n * tl.cdiv(T, BT) * HNAtS + i_hnats
        b_n_iters = tl.load(n_iters_per_block + i_t * HNAtS).to(tl.int32)

        k += (bos * HKV + i_h) * K
        v += (bos * HKV + i_h) * V + i_v * BV
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

        bos_nats, eos_nats = i_n * TNAtS, i_n * TNAtS + TNAtS
        # n_iters_per_block = n_iters_per_block + i_n * tl.cdiv(T, BT) * HATTn_Types * N_types
        # b_n_iters = tl.load(n_iters_per_block + OFFSET_ATTN + i_h_attntypes * N_types).to(tl.int32)
        n_iters_per_block = n_iters_per_block + i_n * tl.cdiv(T, BT) * HNAtS + i_hnats
        b_n_iters = tl.load(n_iters_per_block + i_t * HNAtS).to(tl.int32)

        k += i_n * STRIDE_KB + i_h * K
        v += i_n * STRIDE_VB + i_h * V + i_v * BV
    if STORE_MSK:
        msk += (i_b * HQ + i_hq).to(tl.int64) * T * T

    # TODO This is not the case for nats_block_types, as its TNAtS is typically smaller than the normal T
    # Our current solution is to consider H as 1, i.e., we need to transpose the input data to [B * H, T, 1, D]
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN
    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN

    stride_kt = K * HKV
    stride_vt = V * HKV
    # TODO Check if we really need different types for different heads
    stride_block_types_t = N_TYPES * HNAtS

    offs_t = i_t * BT + tl.arange(0, BT)
    qo_msk = offs_t < T

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # [BT, BV]

    qk_scale = scale
    qk_scale *= 1.44269504  # 1/log(2)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_G:
        g_cumsum += bos * HQ + i_hq
        p_g = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None

    if STAGE & 1:
        if STAGE == 3:
            # b_n_iters = tl.minimum(b_n_iters, tl.cdiv(i_t * BT, BS))
            lo, hi = 0, i_t * BT
        else:
            lo = 0
            hi = T
        # loop over k, v and update accumulator
        # for i_s in tl.range(lo, hi, BS, ):
        for i_iter in tl.range(0, b_n_iters, ):
            b_o_nats_block = tl.load(nats_block_indices + i_iter * stride_block_types_t).to(tl.int32)
            i_s0 = b_o_nats_block * NAtS_BLOCK_SIZE
            for i_iter_nats in range(0, N_CHUNK_PER_NAtS_BLOCK):
                i_s = i_s0 + i_iter_nats * BS
                i_s = tl.multiple_of(i_s, BS)

                p_k = tl.make_block_ptr(k, (K, T), (1, stride_kt), (0, i_s), (BK, BS), (0, 1))
                p_v = tl.make_block_ptr(v, (T, V), (stride_vt, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

                # [BK, BS]
                b_k = tl.load(p_k, boundary_check=(0, 1))
                # [BS, BV]
                b_v = tl.load(p_v, boundary_check=(0, 1))
                # [BT, BS]
                b_s = tl.dot(b_q, b_k) * qk_scale
                if USE_G:
                    o_k = i_s + tl.arange(0, BS)
                    m_k = o_k < T
                    b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
                    b_s += b_gq[:, None] - b_gk[None, :]

                # [BT, BS]
                b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                b_r = exp2(b_mp - b_m)
                # [BT, BS]
                b_p = exp2(b_s - b_m[:, None])
                # [BT]
                b_acc = b_acc * b_r + tl.sum(b_p, 1)
                # [BT, BV]
                b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
                b_mp = b_m
                if STORE_MSK:
                    p_msk = tl.make_block_ptr(msk, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
                    tl.store(p_msk, (tl.zeros([BT, BS], dtype=tl.float32) + 1).to(p_msk.dtype.element_ty),
                             boundary_check=(0,1))

        if (NAtS_BLOCK_SIZE > BT and COMPUTE_PARTIAL_CHUNKS) or STAGE == 1:
            # In this case, there is a chance that we need to compute an incomplete nats chunk
            # For instance, i_T * BT = 16, NATS_BLOCK_SIZE=32, then b_n_iters will be 0, we still need
            # to check if the [0,...16] are valid
            load_idx_chunk = b_n_iters * BS // NAtS_BLOCK_SIZE
            b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
            i_s0 = b_o_nats_block * NAtS_BLOCK_SIZE
            if i_s0 < T:  # meaning that the last block is also attention block, we need to do a further computation
                if STAGE == 1:
                    i_nats_block = i_s0 // NAtS_BLOCK_SIZE
                    is_attn_block = tl.load(nats_block_types + i_nats_block * stride_block_types_t).to(tl.int1)
                    b_n_iters_in_nats_block = tl.where(is_attn_block, tl.cdiv(T - b_n_iters * BS, BS), 0)
                    hi = T
                else:
                    # we start from b_n_iters
                    b_n_iters_in_nats_block = tl.cdiv(i_t * BT - i_s0, BS)
                    hi = i_t * BT

                for i_iter_nats in range(0, b_n_iters_in_nats_block):
                    i_s = i_s0 + i_iter_nats * BS

                    i_s = tl.multiple_of(i_s, BS)

                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_kt), (0, i_s), (BK, BS), (0, 1))
                    p_v = tl.make_block_ptr(v, (T, V), (stride_vt, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

                    # [BK, BS]
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    # [BS, BV]
                    b_v = tl.load(p_v, boundary_check=(0, 1))
                    # [BT, BS]
                    b_s = tl.dot(b_q, b_k) * qk_scale
                    o_k = i_s + tl.arange(0, BS)
                    m_k = o_k < hi

                    if USE_G:
                        p_gk = tl.make_block_ptr(g_cumsum, (T,), (HQ,), (i_s,), (BS,), (0,))
                        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
                        b_s += b_gq[:, None] - b_gk[None, :]

                    # [BT, BS]
                    b_s = tl.where(m_k[None, :], b_s, float('-inf'))
                    b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                    b_r = exp2(b_mp - b_m)
                    # [BT, BS]
                    b_p = exp2(b_s - b_m[:, None])
                    # [BT]
                    b_acc = b_acc * b_r + tl.sum(b_p, 1)
                    # [BT, BV]
                    b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
                    b_mp = b_m

    if STAGE & 2 and (COMPUTE_PARTIAL_CHUNKS or BT > NAtS_BLOCK_SIZE):
        # if we do not need to compute incomplete chunks, and NAtS_BLOCK_SIZE is larger than BT,
        # Then no valid values within (i_t*BT, i_t*BT+BT) will exist, so we can skip them

        if COMPUTE_INCOMPLETE_CHUNK_SCORES:
            b_n_iters = tl.cdiv(min(BT, T - i_t * BT), BS)
        elif USE_SW:
            b_n_iters = tl.cdiv(min(BT + min(i_t*BT, SW_SIZE), T - i_t * BT + SW_SIZE), BS)
        else:
            b_n_iters = tl.cdiv(min(BT, T - i_t * BT) - NAtS_BLOCK_SIZE, BS)
        o_q = i_t * BT + tl.arange(0, BT)

        for i_iter in tl.range(0, b_n_iters, ):
            if USE_SW:
                i_s = max((i_t * BT - SW_SIZE), 0) + i_iter * BS
            else:
                i_s = i_t * BT + i_iter * BS
            i_nats_block = i_s // NAtS_BLOCK_SIZE
            is_attn_block = tl.load(nats_block_types + i_nats_block * stride_block_types_t).to(tl.int1)

            if COMPUTE_PARTIAL_CHUNKS or is_attn_block:
                p_k = tl.make_block_ptr(k, (K, T), (1, stride_kt), (0, i_s), (BK, BS), (1, 0))
                p_v = tl.make_block_ptr(v, (T, V), (stride_vt, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

                # [BS]
                o_k = i_s + tl.arange(0, BS)
                m_k = o_k < T
                # [BK, BS]
                b_k = tl.load(p_k, boundary_check=(0, 1))
                # [BS, BV]
                b_v = tl.load(p_v, boundary_check=(0, 1))
                # [BT, BS]
                b_s = tl.dot(b_q, b_k) * qk_scale

                if USE_G:
                    b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
                    b_s += b_gq[:, None] - b_gk[None, :]

                if COMPUTE_INCOMPLETE_CHUNK_SCORES:
                    m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
                    rows_are_valid = o_q < ((i_nats_block + 1) * NAtS_BLOCK_SIZE)
                    m_s = tl.where(is_attn_block, m_s, m_s & rows_are_valid[:, None])
                elif USE_SW:
                    m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
                    rows_are_valid = o_q < ((i_nats_block + 1) * NAtS_BLOCK_SIZE)
                    m_sw = (o_q[:, None] < o_k[None, :] + SW_SIZE) & m_s
                    m_s = tl.where(is_attn_block, m_s, m_sw)
                else:
                    m_s = (o_q[:, None] >= ((i_nats_block + 1) * NAtS_BLOCK_SIZE)) & m_k[None, :]
                    rows_are_valid = tl.full((BT,), value=1, dtype=tl.int1)
                if STORE_MSK:
                    p_msk = tl.make_block_ptr(msk, (T, T), (T, 1), (i_t * BT, i_s), (BT, BS), (1, 0))
                    tl.store(p_msk, tl.broadcast_to(m_s, [BT, BS]).to(p_msk.dtype.element_ty), boundary_check=(0, 1))

                b_s = tl.where(m_s, b_s, float('-inf'))
                # [BT]
                b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                if COMPUTE_PARTIAL_CHUNKS:
                    rows_are_valid = rows_are_valid or (b_m != float('-inf'))
                    b_r = tl.where(rows_are_valid, exp2(b_mp - b_m), 0)
                    # [BT, BS]
                    b_p = tl.where(rows_are_valid[:, None], exp2(b_s - b_m[:, None]), 0)
                else:
                    b_r = exp2(b_mp - b_m)
                    b_p = exp2(b_s - b_m[:, None])
                # [BT]
                b_acc = b_acc * b_r + tl.sum(b_p, 1)
                # [BT, BV]
                b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
                b_mp = b_m

    # this function remove the invalid
    rows_are_valid = b_m != float('-inf')

    # epilogue
    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)

    b_o = tl.where(rows_are_valid[:, None], b_o, 0.)
    b_m = tl.where(rows_are_valid, b_m, 0.)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_attn_bwd_kernel_preprocess(
        p_o,
        p_do,
        p_delta,
        B: tl.constexpr,
        V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(p_o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(p_do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(p_delta + i_n, b_delta.to(p_delta.dtype.element_ty), )


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BS']),
    'USE_SW': lambda args: args['SW_SIZE'] is not None,
    'COMPUTE_PARTIAL_CHUNKS': lambda args: args['USE_SW'] or args['COMPUTE_INCOMPLETE_CHUNK_SCORES']
})
@triton.jit
def parallel_nats_attn_bwd_kernel_dq(q, k, v,  # Q, O are of shape [B, T, H, DQ]
                                     g_cumsum, lse, delta,
                                     do, dq, dg_cumsum,
                                     nats_block_types, nats_block_indices,  # Both are of shape [B, T, HKV]
                                     n_iters_per_block,
                                     cu_seqlens, chunk_indices, cu_seqlens_nats,
                                     dk, dv, dnats,  # used to store the in complete chunk scores
                                     scale,
                                     T: tl.constexpr,
                                     TNAtS: tl.constexpr,
                                     B: tl.constexpr,
                                     HQ: tl.constexpr,
                                     HKV: tl.constexpr,
                                     G: tl.constexpr,
                                     HNAtS: tl.constexpr,
                                     GNAtS: tl.constexpr,
                                     K: tl.constexpr,
                                     V: tl.constexpr,
                                     BT: tl.constexpr,
                                     BS: tl.constexpr,
                                     BK: tl.constexpr,
                                     BV: tl.constexpr,
                                     NAtS_BLOCK_SIZE: tl.constexpr,
                                     SW_SIZE: tl.constexpr,
                                     N_TYPES: tl.constexpr,
                                     OFFSET_ATTN: tl.constexpr,
                                     COMPUTE_INCOMPLETE_CHUNK_SCORES: tl.constexpr,
                                     STAGE: tl.constexpr,
                                     USE_G: tl.constexpr,
                                     IS_VARLEN: tl.constexpr,
                                     USE_SW: tl.constexpr,
                                     COMPUTE_PARTIAL_CHUNKS: tl.constexpr,
                                     N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
                                     ):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    i_hnats = i_hq // GNAtS

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(
            tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        # In this case we might just assume that each operation contains one dim or
        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
        # TODO we need to check if this is correct!!!
        # b_n_iters = tl.load(n_iters_per_block + i_n * N_types + OFFSET_ATTN).to(tl.int32)

        n_iters_per_block = n_iters_per_block + i_n * tl.cdiv(T, BT) * HNAtS + i_hnats
        b_n_iters = tl.load(n_iters_per_block + i_t * HNAtS).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T

        bos_nats, eos_nats = i_n * TNAtS, i_n * TNAtS + TNAtS
        # n_iters_per_block = n_iters_per_block + i_n * tl.cdiv(T, BT) * HATTn_Types * N_types
        # b_n_iters = tl.load(n_iters_per_block + OFFSET_ATTN + i_h_attntypes * N_types).to(tl.int32)
        n_iters_per_block = n_iters_per_block + i_n * tl.cdiv(T, BT) * HNAtS + i_hnats
        b_n_iters = tl.load(n_iters_per_block + i_t * HNAtS).to(tl.int32)

    k += (bos * HKV + i_h) * K
    v += (bos * HKV + i_h) * V

    # TODO This is not the case for nats_block_types, as its TNAtS is typically smaller than the normal T
    # Our current solution is to consider H as 1, i.e., we need to transpose the input data to [B * H, T, 1, D]
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN
    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN

    stride_kt = K * HKV
    stride_vt = V * HKV
    # TODO Check if we really need different types for different heads
    stride_block_types_t = N_TYPES * HNAtS
    stride_qt = K * HQ
    stride_ot = V * HQ
    stride_lset = HQ

    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    qk_scale = scale * 1.44269504  # 1/log(2)

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # [BT, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BT]
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros([BT, ], dtype=tl.float32)
        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None
        b_dg = None

    if COMPUTE_PARTIAL_CHUNKS:
        dk += (bos * HQ + i_hq) * K
        dv += (bos * HQ + i_hq) * V

    if STAGE & 1:
        if STAGE == 3:
            lo, hi = 0, i_t * BT
        else:
            lo, hi = 0, T
        for i_iter in tl.range(0, b_n_iters, ):
            b_o_nats_block = tl.load(nats_block_indices + i_iter * stride_block_types_t).to(tl.int32)
            i_s0 = b_o_nats_block * NAtS_BLOCK_SIZE
            for i_iter_nats in range(0, N_CHUNK_PER_NAtS_BLOCK):
                i_s = i_s0 + i_iter_nats * BS
                i_s = tl.multiple_of(i_s, BS)

                p_k = tl.make_block_ptr(k, (K, T), (1, stride_kt), (0, i_s), (BK, BS), (0, 1))
                p_v = tl.make_block_ptr(v, (V, T), (1, stride_vt), (i_v * BV, i_s), (BV, BS), (1, 0))
                o_k = i_s + tl.arange(0, BS)
                m_k = o_k < T

                # [BK, BS]
                b_k = tl.load(p_k, boundary_check=(0, 1))
                # [BV, BS]
                b_v = tl.load(p_v, boundary_check=(0, 1))
                # [BT, BS]
                b_s = tl.dot(b_q, b_k) * qk_scale
                if USE_G:
                    b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
                    b_s += b_gq[:, None] - b_gk[None, :]

                b_p = exp2(b_s - b_lse[:, None])
                b_dp = tl.dot(b_do, b_v)
                b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
                b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
                if USE_G:
                    b_dg += tl.sum(b_ds, 1)
            if (NAtS_BLOCK_SIZE > BT and COMPUTE_PARTIAL_CHUNKS) or STAGE == 1:
                # In this case, there is a chance that we need to compute an incomplete nats chunk
                # For instance, i_T * BT = 16, NATS_BLOCK_SIZE=32, then b_n_iters will be 0, we still need
                # to check if the [0,...16] are valid
                load_idx_chunk = b_n_iters * BS // NAtS_BLOCK_SIZE
                b_o_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
                i_s0 = b_o_nats_block * NAtS_BLOCK_SIZE
                if i_s0 < T:  # meaning that the last block is also attention block, we need to do a further computation
                    i_nats_block = i_s0 // NAtS_BLOCK_SIZE
                    is_attn_block = tl.load(nats_block_types + i_nats_block * stride_block_types_t).to(tl.int1)

                    if STAGE == 1:
                        b_n_iters_in_nats_block = tl.where(is_attn_block, tl.cdiv(T - b_n_iters * BS, BS), 0)
                        hi = T
                    else:
                        # we start from b_n_iters
                        b_n_iters_in_nats_block = tl.cdiv(i_t * BT - i_s0, BS)
                        hi = i_t * BT

                    for i_iter_nats in range(0, b_n_iters_in_nats_block):
                        i_s = i_s0 + i_iter_nats * BS

                        i_s = tl.multiple_of(i_s, BS)
                        p_k = tl.make_block_ptr(k, (K, T), (1, stride_kt), (0, i_s), (BK, BS), (0, 1))
                        p_v = tl.make_block_ptr(v, (V, T), (1, stride_vt), (i_v * BV, i_s), (BV, BS), (1, 0))

                        # [BK, BS]
                        b_k = tl.load(p_k, boundary_check=(0, 1))
                        # [BS, BV]
                        b_v = tl.load(p_v, boundary_check=(0, 1))
                        # [BT, BS]
                        b_s = tl.dot(b_q, b_k) * qk_scale
                        if USE_G:
                            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
                            b_s += b_gq[:, None] - b_gk[None, :]

                        b_p = exp2(b_s - b_lse[:, None])
                        b_dp = tl.dot(b_do, b_v)
                        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
                        if COMPUTE_PARTIAL_CHUNKS and (not is_attn_block):
                            # we also need to compute the non-nats attention blocks if we compute incomplete blocks with
                            # attention as they will not be considered during dkdv
                            # TODO we need to test these !!!
                            b_dv = tl.dot(tl.trans(b_p).to(b_do.dtype), b_do)
                            b_dk = tl.dot(tl.trans(b_ds).to(b_q.dtype), b_q) * scale

                            p_dk = tl.make_block_ptr(dk, (T, K), (stride_kt, 1), (i_s, 0), (BS, BK), (1, 0))
                            p_dv = tl.make_block_ptr(dv, (T, V), (stride_vt, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

                            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0,1))
                            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0,1))

                        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
                        if USE_G:
                            b_dg += tl.sum(b_ds, 1)

        if STAGE & 2 and (COMPUTE_PARTIAL_CHUNKS or BT > NAtS_BLOCK_SIZE):
            # if we do not need to compute incomplete chunks, and NAtS_BLOCK_SIZE is larger than BT,
            # Then no valid values within (i_t*BT, i_t*BT+BT) will exist, so we can skip them

            if COMPUTE_INCOMPLETE_CHUNK_SCORES:
                b_n_iters = tl.cdiv(min(BT, T - i_t * BT), BS)
            elif USE_SW:
                b_n_iters = tl.cdiv(min(BT + min(i_t * BT, SW_SIZE), T - i_t * BT + SW_SIZE), BS)
            else:
                b_n_iters = tl.cdiv(min(BT, T - i_t * BT) - NAtS_BLOCK_SIZE, BS)

            o_q = i_t * BT + tl.arange(0, BT)
            for i_iter in tl.range(0, b_n_iters, ):
                if USE_SW:
                    i_s = max((i_t * BT - SW_SIZE), 0) + i_iter * BS
                else:
                    i_s = i_t * BT + i_iter * BS
                i_nats_block = i_s // NAtS_BLOCK_SIZE
                is_attn_block = tl.load(nats_block_types + i_nats_block * stride_block_types_t).to(tl.int1)

                #if is_attn_block or COMPUTE_INCOMPLETE_CHUNK_SCORES:
                if COMPUTE_PARTIAL_CHUNKS or is_attn_block:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_kt), (0, i_s), (BK, BS), (0, 1))
                    p_v = tl.make_block_ptr(v, (V, T), (1, stride_vt), (i_v * BV, i_s), (BV, BS), (1, 0))

                    # [BS]
                    o_k = i_s + tl.arange(0, BS)
                    m_k = o_k < T
                    # [BK, BS]
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    # [BS, BV]
                    b_v = tl.load(p_v, boundary_check=(0, 1))
                    # [BT, BS]
                    b_s = tl.dot(b_q, b_k) * qk_scale

                    if USE_G:
                        p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
                        b_s += b_gq[:, None] - b_gk[None, :]

                    if COMPUTE_INCOMPLETE_CHUNK_SCORES:
                        #m_s_lower_inverse = (o_q[:, None] < ((i_nats_block + 1) * NAtS_BLOCK_SIZE))
                        m_s = (o_q[:, None] >= o_k[None, :])
                        rows_are_valid = o_q < ((i_nats_block + 1) * NAtS_BLOCK_SIZE)
                        m_s = tl.where(is_attn_block, m_s, m_s & rows_are_valid[:, None]) & m_k[None, :]
                    elif USE_SW:
                        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
                        m_sw = ((o_q[:, None] < o_k[None, :] + SW_SIZE) & m_s) #& rows_are_valid[:, None]
                        m_s = tl.where(is_attn_block, m_s, m_sw) & m_k[None, :]
                    else:
                        m_s_lower = (o_q[:, None] >= ((i_nats_block + 1) * NAtS_BLOCK_SIZE))
                        m_s = m_s_lower & m_k[None, :]

                    b_p = tl.where(m_s, exp2(b_s - b_lse[:, None]), 0)
                    # [BT, BV] @ [BV, BS] -> [BT, BS]
                    b_dp = tl.dot(b_do, b_v)
                    b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
                    # [BT, BS] @ [BS, BK] -> [BT, BK]
                    b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
                    if COMPUTE_INCOMPLETE_CHUNK_SCORES and not is_attn_block:

                        # we also need to compute the non-nats attention blocks if we compute incomplete blocks with
                        # attention as they will not be considered during dkdv
                        b_dv = tl.dot(tl.trans(b_p).to(b_do.dtype), b_do)
                        b_dk = tl.dot(tl.trans(b_ds).to(b_q.dtype), b_q) * scale

                        p_dk = tl.make_block_ptr(dk, (T, K), (stride_qt, 1), (i_s, 0), (BS, BK), (1, 0))
                        p_dv = tl.make_block_ptr(dv, (T, V), (stride_ot, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
                        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty))
                        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty))
                    if USE_G:
                        # TOOD check if we need to consider COMPUTE_INCOMPLETE_CHUNK_SCORES and not seperatedly...
                        b_dg += tl.sum(b_ds, 1)

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT']),
    'USE_SW': lambda args: args['SW_SIZE'] is not None,
    'COMPUTE_PARTIAL_CHUNKS': lambda args: args['USE_SW'] or args['COMPUTE_INCOMPLETE_CHUNK_SCORES']
})
@triton.jit
def parallel_attn_bwd_kernel_dkdv(q, k, v,  # Q, O are of shape [B, T, H, DQ]
                                  g_cumsum, lse, delta,
                                  do, dg_cumsum,
                                  nats_block_types, nats_block_indices,  # Both are of shape [B, T, HAttns]
                                  chunk_indices_attn_nats, cu_seqlens, chunk_indices, cu_seqlens_nats,
                                  dk, dv, dnats,  # used to store the in complete chunk scores
                                  scale,
                                  T: tl.constexpr,
                                  TNAtS: tl.constexpr,
                                  B: tl.constexpr,
                                  HQ: tl.constexpr,
                                  HKV: tl.constexpr,
                                  G: tl.constexpr,
                                  HNAtS: tl.constexpr,
                                  GNAtS: tl.constexpr,
                                  K: tl.constexpr,
                                  V: tl.constexpr,
                                  BK: tl.constexpr,
                                  BV: tl.constexpr,
                                  BT: tl.constexpr,
                                  BS: tl.constexpr,
                                  NAtS_BLOCK_SIZE: tl.constexpr,
                                  SW_SIZE: tl.constexpr,
                                  N_TYPES: tl.constexpr,
                                  OFFSET_ATTN: tl.constexpr,
                                  COMPUTE_INCOMPLETE_CHUNK_SCORES: tl.constexpr,
                                  STAGE: tl.constexpr,
                                  USE_G: tl.constexpr,
                                  IS_VARLEN: tl.constexpr,
                                  N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
                                  USE_SW: tl.constexpr,
                                  COMPUTE_PARTIAL_CHUNKS: tl.constexpr,
                                  ):
    # i_v, off_n, off_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_v, i_t_, i_gnats = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t_ // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t_ % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t_
        i_t_nats_offset = 0

    off_bh_nats = tl.load(chunk_indices_attn_nats + i_t_ * 2).to(tl.int32)
    i_t = tl.load(chunk_indices_attn_nats + i_t_ * 2 + 1).to(tl.int32)
    i_b = off_bh_nats // HNAtS
    i_hnats = off_bh_nats % HNAtS

    i_hq = i_hnats * GNAtS + i_gnats
    i_h = i_hq // G

    if IS_VARLEN:
        # i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        # i_n = off_b
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

    q += (bos * HQ + i_hq) * K
    lse += bos * HQ + i_hq

    do += (bos * HQ + i_hq) * V
    delta += bos * HQ + i_hq

    # TODO This is not the case for nats_block_types, as its TNAtS is typically smaller than the normal T
    # Our current solution is to consider H as 1, i.e., we need to transpose the input data to [B * H, T, 1, D]
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN
    dnats += (bos_nats * N_CHUNK_PER_NAtS_BLOCK * HQ + i_hq)
    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN

    stride_kt = K * HKV
    stride_vt = V * HKV

    stride_dkt = K * HQ
    stride_dvt = V * HQ
    # TODO Check if we really need different types for different heads
    stride_block_types_t = N_TYPES * HNAtS
    stride_dblock_attn_t = HQ
    stride_qt = K * HQ
    stride_ot = V * HQ
    stride_lset = HQ

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    b_i_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
    i_t0 = b_i_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT

    # we only check the nats_chunks that are valid

    qk_scale = scale * 1.44269504  # 1/log(2)

    # If it is Stage 1, for values greater than hi, they are involved in STAGE 2

    p_k = tl.make_block_ptr(k + (bos * HKV + i_h) * K, (T, K), (HKV * K, 1), (i_t0, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * HKV + i_h) * V, (T, V), (HKV * V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t0, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))

    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))

    o_k = i_t0 + tl.arange(0, BT)
    m_k = o_k < T

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    b_dattn = 0.

    if USE_G:
        p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t0,), (BT,), (0,))
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_dg = tl.zeros([BT, ], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None

    if STAGE & 2 and COMPUTE_PARTIAL_CHUNKS:
        lo, hi = i_t0, min(i_t0 + BT, T)

        for i_s in range(lo, hi, BS):
            p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
            p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

            # [BS]
            o_q = i_s + tl.arange(0, BS)
            m_q = o_q < T
            # [BS, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BS, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BS]
            b_lse = tl.load(p_lse, boundary_check=(0,))
            b_delta = tl.load(p_delta, boundary_check=(0,))
            # [BT, BS]
            b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
            if USE_G:
                p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                b_s += b_gq[None, :] - b_gk[:, None]
            b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
            # [BT, BS] @ [BS, BV] -> [BT, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BT, BV] @ [BV, BS] -> [BT, BS]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BT, BS]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BT, BS] @ [BS, BK] -> [BT, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
            if USE_G:
                b_dg -= tl.sum(b_ds, 1)

    if STAGE & 1:
        if STAGE == 3:
            if COMPUTE_PARTIAL_CHUNKS:
                lo = i_t0 + BT
            else:
                lo = b_i_nats_block * NAtS_BLOCK_SIZE + NAtS_BLOCK_SIZE
        else:
            lo = 0
        n_iters = (T - lo) // BS
        for i_s_ in range(n_iters):
            i_s = lo + i_s_ * BS
            i_s = tl.multiple_of(i_s, BS)
            p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
            p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

            # [BS]
            o_q = i_s + tl.arange(0, BS)
            m_q = o_q < T
            # [BS, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BS, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BS]
            b_lse = tl.load(p_lse, boundary_check=(0,))
            b_delta = tl.load(p_delta, boundary_check=(0,))
            # [BT, BS]
            b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
            if USE_G:
                p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                b_s += b_gq[None, :] - b_gk[:, None]
            b_p = exp2(b_s - b_lse[None, :])
            # [BT, BS] @ [BS, BV] -> [BT, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BT, BV] @ [BV, BS] -> [BT, BS]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BT, BS]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BT, BS] @ [BS, BK] -> [BT, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

            b_dattn += tl.sum(tl.where(m_k[:, None], b_ds, 0))

            if USE_G:
                b_dg -= tl.sum(b_ds, 1)
        hi = tl.cdiv(T - lo, BS) * BS + lo
        if hi > T:
            i_s = hi - BS
            # the last block with q partially involved
            p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
            p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

            # [BS]
            o_q = i_s + tl.arange(0, BS)
            m_q = o_q < T
            # [BS, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BS, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BS]
            b_lse = tl.load(p_lse, boundary_check=(0,))
            b_delta = tl.load(p_delta, boundary_check=(0,))
            # [BT, BS]
            b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
            if USE_G:
                p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                b_s += b_gq[None, :] - b_gk[:, None]
            b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
            # [BT, BS] @ [BS, BV] -> [BT, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BT, BV] @ [BV, BS] -> [BT, BS]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BT, BS]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BT, BS] @ [BS, BK] -> [BT, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

            b_dattn += tl.sum(tl.where(m_k[:, None], b_ds, 0))

            if USE_G:
                b_dg -= tl.sum(b_ds, 1)

    b_dk = b_dk * scale
    # we also need to read valuesfrom dk dv
    #if COMPUTE_INCOMPLETE_CHUNK_SCORES:
    #    b_dk += tl.load(p_dk, boundary_check=(0,1))
    #    b_dv += tl.load(p_dv, boundary_check=(0,1))


    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    # TODO currently we only consider the gradients where the blocks are selected as attns.
    #  We could also in principle compute
    tl.store(
        dnats + (b_i_nats_block * N_CHUNK_PER_NAtS_BLOCK + i_t_nats_offset) * stride_dblock_attn_t,
        b_dattn.to(dnats.dtype.element_ty)
    )

    if USE_G:
        p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))



@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'N_CHUNK_PER_NAtS_BLOCK': lambda args: triton.cdiv(args['NAtS_BLOCK_SIZE'], args['BT']),
    'USE_SW': lambda args: args['SW_SIZE'] is not None,
    'COMPUTE_PARTIAL_CHUNKS': lambda args: args['USE_SW'] or args['COMPUTE_INCOMPLETE_CHUNK_SCORES']
})
@triton.jit
def parallel_attn_bwd_kernel_dkdv_within_valid_blocks(q, k, v,  # Q, O are of shape [B, T, H, DQ]
                                                      g_cumsum, lse, delta,
                                                      do, dg_cumsum,
                                                      nats_block_types, nats_block_indices,  # Both are of shape [B, T, HAttns]
                                                      cu_seqlens, cu_seqlens_nats,
                                                      dk, dv, dnats,  # used to store the in complete chunk scores
                                                      scale,
                                                      T: tl.constexpr,
                                                      TNAtS: tl.constexpr,
                                                      B: tl.constexpr,
                                                      HQ: tl.constexpr,
                                                      HKV: tl.constexpr,
                                                      G: tl.constexpr,
                                                      HNAtS: tl.constexpr,
                                                      GNAtS: tl.constexpr,
                                                      K: tl.constexpr,
                                                      V: tl.constexpr,
                                                      BK: tl.constexpr,
                                                      BV: tl.constexpr,
                                                      BT: tl.constexpr,
                                                      BS: tl.constexpr,
                                                      NAtS_BLOCK_SIZE: tl.constexpr,
                                                      SW_SIZE:tl.constexpr,
                                                      N_TYPES: tl.constexpr,
                                                      OFFSET_ATTN: tl.constexpr,
                                                      COMPUTE_INCOMPLETE_CHUNK_SCORES: tl.constexpr,
                                                      COMPUTE_DNATS_FOR_INVALID_BLCOKS: tl.constexpr,
                                                      STAGE: tl.constexpr,
                                                      USE_G: tl.constexpr,
                                                      IS_VARLEN: tl.constexpr,
                                                      N_CHUNK_PER_NAtS_BLOCK: tl.constexpr,
                                                      USE_SW:tl.constexpr,
                                                      COMPUTE_PARTIAL_CHUNKS: tl.constexpr,
                                                      ):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    #i_v, i_t_, i_gnats = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ

    if N_CHUNK_PER_NAtS_BLOCK > 1:
        i_t_nats = i_t // N_CHUNK_PER_NAtS_BLOCK
        i_t_nats_offset = i_t % N_CHUNK_PER_NAtS_BLOCK
    else:
        i_t_nats = i_t
        i_t_nats_offset = 0

    i_hnats = i_hq // GNAtS
    i_gnats = i_hq % GNAtS
    i_h = i_hq // G

    if IS_VARLEN:
        # i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_n).to(tl.int32), tl.load(cu_seqlens_nats + i_n + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        # i_n = off_b
        bos, eos = i_b * T, i_b * T + T
        bos_nats, eos_nats = i_b * TNAtS, i_b * TNAtS + TNAtS

    q += (bos * HQ + i_hq) * K
    lse += bos * HQ + i_hq

    do += (bos * HQ + i_hq) * V
    delta += bos * HQ + i_hq

    # TODO This is not the case for nats_block_types, as its TNAtS is typically smaller than the normal T
    # Our current solution is to consider H as 1, i.e., we need to transpose the input data to [B * H, T, 1, D]
    dnats += (bos_nats * N_CHUNK_PER_NAtS_BLOCK * HQ + i_hq)
    nats_block_indices += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN
    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_ATTN

    stride_kt = K * HKV
    stride_vt = V * HKV

    stride_dkt = K * HQ
    stride_dvt = V * HQ
    # TODO Check if we really need different types for different heads
    stride_block_types_t = N_TYPES * HNAtS
    stride_dblock_attn_t = HQ
    stride_qt = K * HQ
    stride_ot = V * HQ
    stride_lset = HQ


    #load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    #b_i_nats_block = tl.load(nats_block_indices + load_idx_chunk * stride_block_types_t).to(tl.int32)
    #i_t0 = b_i_nats_block * NAtS_BLOCK_SIZE + i_t_nats_offset * BT
    i_t0 = i_t * BT

    # we only check the nats_chunks that are valid

    qk_scale = scale * 1.44269504  # 1/log(2)

    # If it is Stage 1, for values greater than hi, they are involved in STAGE 2

    p_k = tl.make_block_ptr(k + (bos * HKV + i_h) * K, (T, K), (HKV * K, 1), (i_t0, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * HKV + i_h) * V, (T, V), (HKV * V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))

    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))

    o_k = i_t0 + tl.arange(0, BT)
    m_k = o_k < T

    b_dattn = 0.

    if USE_G:
        p_gk = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t0,), (BT,), (0,))
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        b_dg = tl.zeros([BT, ], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None

    load_idx_chunk = i_t * BT // NAtS_BLOCK_SIZE
    chunk_is_attn = tl.load(nats_block_types + load_idx_chunk * stride_block_types_t).to(tl.int1)
    if chunk_is_attn:
        p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t0, 0), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t0, i_v * BV), (BT, BV), (1, 0))

        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)

        if STAGE & 2 and COMPUTE_PARTIAL_CHUNKS:
            lo, hi = i_t0, min(i_t0 + BT, T)

            for i_s in range(lo, hi, BS):
                p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
                p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
                p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
                p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

                # [BS]
                o_q = i_s + tl.arange(0, BS)
                m_q = o_q < T
                # [BS, BK]
                b_q = tl.load(p_q, boundary_check=(0, 1))
                # [BS, BV]
                b_do = tl.load(p_do, boundary_check=(0, 1))
                # [BS]
                b_lse = tl.load(p_lse, boundary_check=(0,))
                b_delta = tl.load(p_delta, boundary_check=(0,))
                # [BT, BS]
                b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
                if USE_G:
                    p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                    b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                    b_s += b_gq[None, :] - b_gk[:, None]
                b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
                # [BT, BS] @ [BS, BV] -> [BT, BV]
                b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
                # [BT, BV] @ [BV, BS] -> [BT, BS]
                b_dp = tl.dot(b_v, tl.trans(b_do))
                # [BT, BS]
                b_ds = b_p * (b_dp - b_delta[None, :])
                # [BT, BS] @ [BS, BK] -> [BT, BK]
                b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
                if USE_G:
                    b_dg -= tl.sum(b_ds, 1)

        if STAGE & 1:
            if STAGE == 3:
                if COMPUTE_PARTIAL_CHUNKS:
                    lo = i_t0 + max(BT, BS)
                else:
                    lo = load_idx_chunk * NAtS_BLOCK_SIZE + NAtS_BLOCK_SIZE
            else:
                lo = 0
            #n_iters = (T - lo) // BS
            n_iters = tl.cdiv(T-lo, BS)
            for i_s_ in range(n_iters):
                i_s = lo + i_s_ * BS
                #i_s = tl.multiple_of(i_s, BS)
                p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
                p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
                p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
                p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

                # [BS]
                o_q = i_s + tl.arange(0, BS)
                m_q = o_q < T
                # [BS, BK]
                b_q = tl.load(p_q, boundary_check=(0, 1))
                # [BS, BV]
                b_do = tl.load(p_do, boundary_check=(0, 1))
                # [BS]
                b_lse = tl.load(p_lse, boundary_check=(0,))
                b_delta = tl.load(p_delta, boundary_check=(0,))
                # [BT, BS]
                b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
                if USE_G:
                    p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                    b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                    b_s += b_gq[None, :] - b_gk[:, None]
                b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
                # [BT, BS] @ [BS, BV] -> [BT, BV]
                b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
                # [BT, BV] @ [BV, BS] -> [BT, BS]
                b_dp = tl.dot(b_v, tl.trans(b_do))
                # [BT, BS]
                b_ds = b_p * (b_dp - b_delta[None, :])
                # [BT, BS] @ [BS, BK] -> [BT, BK]
                b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

                b_dattn += tl.sum(tl.where(m_k[:, None], b_ds, 0))

                if USE_G:
                    b_dg -= tl.sum(b_ds, 1)
            """
            hi = tl.cdiv(T - lo, BS) * BS + lo
            if hi > T and lo < T:
                i_s = hi - BS

                # the last block with q partially involved
                p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
                p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
                p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
                p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

                # [BS]
                o_q = i_s + tl.arange(0, BS)
                m_q = o_q < T
                # [BS, BK]
                b_q = tl.load(p_q, boundary_check=(0, 1))
                # [BS, BV]
                b_do = tl.load(p_do, boundary_check=(0, 1))
                # [BS]
                b_lse = tl.load(p_lse, boundary_check=(0,))
                b_delta = tl.load(p_delta, boundary_check=(0,))
                # [BT, BS]
                b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
                if USE_G:
                    p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                    b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                    b_s += b_gq[None, :] - b_gk[:, None]
                b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
                # [BT, BS] @ [BS, BV] -> [BT, BV]
                b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
                # [BT, BV] @ [BV, BS] -> [BT, BS]
                b_dp = tl.dot(b_v, tl.trans(b_do))
                # [BT, BS]
                b_ds = b_p * (b_dp - b_delta[None, :])
                # [BT, BS] @ [BS, BK] -> [BT, BK]
                b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
                # TODO consider the impact of sliding window sizes for dnats?
                b_dattn += tl.sum(tl.where(m_k[:, None], b_ds, 0))

                if USE_G:
                    b_dg -= tl.sum(b_ds, 1)
            """

        b_dk = b_dk * scale
        # we also need to read valuesfrom dk dv
        #if COMPUTE_INCOMPLETE_CHUNK_SCORES:
        #    b_dk += tl.load(p_dk, boundary_check=(0,1))
        #    b_dv += tl.load(p_dv, boundary_check=(0,1))

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # TODO currently we only consider the gradients where the blocks are selected as attns.
        #  We could also in principle compute
        tl.store(
            dnats + i_t * stride_dblock_attn_t,
            b_dattn.to(dnats.dtype.element_ty)
        )

        if USE_G:
            p_dg = tl.make_block_ptr(dg_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    else:
        # THis is the case where we only compute d_nats. This might be too expensive, we need to check if we could
        # approximate this gradient information...
        if STAGE & 1:
            if USE_SW:
                # WE first need to compute dk dv for teh sliding window sizes

                b_dk = tl.zeros([BT, BK], dtype=tl.float32)
                b_dv = tl.zeros([BT, BV], dtype=tl.float32)

                n_iters1 = tl.cdiv(min((BT + SW_SIZE), T - BT), BS)
                lo, hi = i_t0, min(i_t0 + BT, T)
                for i_s_ in range(n_iters1):
                    i_s = lo + i_s_ * BS
                    i_s = tl.multiple_of(i_s, BS)
                    p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
                    p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
                    p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
                    p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

                    # [BS]
                    o_q = i_s + tl.arange(0, BS)
                    m_q = o_q < T
                    # [BS, BK]
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    # [BS, BV]
                    b_do = tl.load(p_do, boundary_check=(0, 1))
                    # [BS]
                    b_lse = tl.load(p_lse, boundary_check=(0,))
                    b_delta = tl.load(p_delta, boundary_check=(0,))
                    # [BT, BS]
                    b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
                    if USE_G:
                        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                        b_s += b_gq[None, :] - b_gk[:, None]
                    m_s = (o_k[:, None] <= o_q[None, :])
                    m_sw = ((o_q[None, :] < o_k[:, None] + SW_SIZE) & m_s)

                    b_p = tl.where(m_sw & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
                    b_dp = tl.dot(b_v, tl.trans(b_do))
                    b_ds = b_p * (b_dp - b_delta[None, :])
                    if USE_SW:
                        # [BT, BS] @ [BS, BV] -> [BT, BV]
                        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
                        # [BT, BV] @ [BV, BS] -> [BT, BS]
                        # [BT, BS]
                        # [BT, BS] @ [BS, BK] -> [BT, BK]
                        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)
                    if COMPUTE_DNATS_FOR_INVALID_BLCOKS:
                        b_p1 = tl.where(m_s & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
                        b_p1 = tl.clamp(b_p1, 0., 1.)
                        b_ds1 = b_p1 * (b_dp - b_delta[None, :])
                        b_dattn += tl.sum(
                            tl.where(m_k[:, None] & o_q > load_idx_chunk * NAtS_BLOCK_SIZE + NAtS_BLOCK_SIZE, b_ds1, 0)
                        )
                    if USE_G:
                        b_dg -= tl.sum(b_ds, 1)
            if USE_SW:
                p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t0, 0), (BT, BK), (1, 0))
                p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t0, i_v * BV), (BT, BV),
                                         (1, 0))
                b_dk = b_dk * scale
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
            # Now we need to work on blocks without any attn scores
            if COMPUTE_DNATS_FOR_INVALID_BLCOKS:

                if STAGE == 3:
                    #lo = load_idx_chunk * NAtS_BLOCK_SIZE + NAtS_BLOCK_SIZE + SW_SIZE
                    if USE_SW:
                        lo = i_t0 + n_iters1 * BT
                    else:
                        lo = load_idx_chunk * NAtS_BLOCK_SIZE + NAtS_BLOCK_SIZE + SW_SIZE
                else:
                    lo = 0
                n_iters = (T - lo) // BS
                for i_s_ in range(n_iters):
                    i_s = lo + i_s_ * BS
                    i_s = tl.multiple_of(i_s, BS)
                    p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
                    p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
                    p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
                    p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

                    # [BS]
                    o_q = i_s + tl.arange(0, BS)
                    # [BS, BK]
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    # [BS, BV]
                    b_do = tl.load(p_do, boundary_check=(0, 1))
                    # [BS]
                    b_lse = tl.load(p_lse, boundary_check=(0,))
                    b_delta = tl.load(p_delta, boundary_check=(0,))
                    # [BT, BS]
                    b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
                    if USE_G:
                        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                        b_s += b_gq[None, :] - b_gk[:, None]
                    b_p = exp2(b_s - b_lse[None, :])
                    b_p = tl.clamp(b_p, 0., 1.)
                    # [BT, BS] @ [BS, BV] -> [BT, BV]
                    # [BT, BV] @ [BV, BS] -> [BT, BS]
                    b_dp = tl.dot(b_v, tl.trans(b_do))
                    # [BT, BS]
                    b_ds = b_p * (b_dp - b_delta[None, :])
                    # [BT, BS] @ [BS, BK] -> [BT, BK]
                    b_dattn += tl.sum(tl.where(m_k[:, None], b_ds, 0))
                hi = tl.cdiv(T - lo, BS) * BS + lo
                if hi > T:
                    i_s = hi - BS
                    # the last block with q partially involved
                    p_q = tl.make_block_ptr(q, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0))
                    p_do = tl.make_block_ptr(do, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
                    p_lse = tl.make_block_ptr(lse, (T,), (HQ,), (i_s,), (BS,), (0,))
                    p_delta = tl.make_block_ptr(delta, (T,), (HQ,), (i_s,), (BS,), (0,))

                    # [BS]
                    o_q = i_s + tl.arange(0, BS)
                    m_q = o_q < T
                    # [BS, BK]
                    b_q = tl.load(p_q, boundary_check=(0, 1))
                    # [BS, BV]
                    b_do = tl.load(p_do, boundary_check=(0, 1))
                    # [BS]
                    b_lse = tl.load(p_lse, boundary_check=(0,))
                    b_delta = tl.load(p_delta, boundary_check=(0,))
                    # [BT, BS]
                    b_s = tl.dot(b_k, tl.trans(b_q)) * qk_scale
                    if USE_G:
                        p_gq = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
                        b_gq = tl.load(p_gq, boundary_check=(0,)).to(tl.float32)
                        b_s += b_gq[None, :] - b_gk[:, None]
                    b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
                    b_p = tl.clamp(b_p, 0., 1.)
                    # [BT, BS] @ [BS, BV] -> [BT, BV]
                    # [BT, BV] @ [BV, BS] -> [BT, BS]
                    b_dp = tl.dot(b_v, tl.trans(b_do))
                    # [BT, BS]
                    b_ds = b_p * (b_dp - b_delta[None, :])
                    # [BT, BS] @ [BS, BK] -> [BT, BK]
                    b_dattn += tl.sum(tl.where(m_k[:, None], b_ds, 0))

                # TODO currently we only consider the gradients where the blocks are selected as attns.
                #  We could also in principle compute
                tl.store(
                    dnats + i_t * stride_dblock_attn_t,
                    b_dattn.to(dnats.dtype.element_ty)
                )


def parallel_attn_nats_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.IntTensor,
        g_cumsum: torch.Tensor,
        scale: float,
        NAtS_block_size: int,
        offset_attn: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        chunk_size: int = 128,
        sliding_window_size: int | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_nats: torch.LongTensor | None = None,
        is_causal=True,
        store_msk: bool = False,
):
    B, T , H, K, V = *k.shape, v.shape[-1]

    TNAtS, HNAtS, N_TYPES = nats_block_types.shape[1:]
    HQ = q.shape[2]
    G = HQ // H
    GNAtS = HQ // HNAtS

    BT = chunk_size

    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
        num_warps = 8
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
        num_warps = 4
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
        num_warps = 2
    assert NAtS_block_size >= BS

    if BV < V:
        BV = V

    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert NK == 1, "The key dimension can not be larger than 256"
    if sliding_window_size is not None:
        assert sliding_window_size >= NAtS_block_size, 'The sliding window size must be greater than the nats block size!'

    n_iters_per_block = compute_attn_n_iters_per_block(
        nats_block_indices,
        q.shape[1],
        BT,
        BS,
        NAtS_block_size,
        offset_attn,
        0,
        sliding_window_size,
    )

    stage = 3 if is_causal else 1
    # TODO merge all evaluation types into one single function!
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    grid = (NV, NT, B * HQ)
    # """
    if store_msk:
        msk = torch.zeros(B, HQ, T, T, dtype=torch.float32, device=q.device)
    else:
        msk = None

    parallel_nats_attn_fwd_kernel[grid](
        q=q, k=k, v=v, o=o, msk=msk,
        g_cumsum=g_cumsum, lse=lse,
        nats_block_types=nats_block_types, nats_block_indices=nats_block_indices,
        n_iters_per_block=n_iters_per_block,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, cu_seqlens_nats=cu_seqlens_nats,
        scale=scale, T=T, TNAtS=TNAtS,
        B=B,
        HQ=HQ,
        HKV=H,
        G=G,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        STRIDE_KB=k.stride(0),
        STRIDE_VB=v.stride(0),
        NAtS_BLOCK_SIZE=NAtS_block_size,
        SW_SIZE=sliding_window_size,
        STAGE=stage,
        COMPUTE_INCOMPLETE_CHUNK_SCORES=compute_incomplete_chunk_scores,
        N_TYPES=N_TYPES,
        OFFSET_ATTN=offset_attn,
        STORE_MSK=store_msk,
        num_warps=num_warps,
    )

    return o, lse, msk


def parallel_attn_bwd_preprocess(
        o: torch.Tensor,
        do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[(delta.numel(),)](
        p_o=o,
        p_do=do,
        p_delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_attn_nats_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.LongTensor,
        n_nats_blocks: torch.LongTensor,
        g_cumsum: torch.Tensor,
        lse: torch.Tensor,
        do: torch.Tensor,
        scale: float = None,
        NAtS_block_size: int = 64,
        OFFSET_ATTN: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        compute_dnats_for_invalid_blocks_attn: bool = False,
        sliding_window_size: int | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_nats: torch.LongTensor | None = None,
        is_causal=True
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    TNAtS, HNAtS, N_TYPES = nats_block_types.shape[1:]

    assert NAtS_block_size >= 64

    G = HQ // H
    GNAtS = HQ // HNAtS

    if check_shared_mem('hopper'):
        BT = 128
        BS = 64
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)
        num_warps = 8
    elif check_shared_mem('ampere'):
        BS = 32
        BK = triton.next_power_of_2(K)
        BV = triton.next_power_of_2(V)
        BT = min(128 if K <= 64 else 64, NAtS_block_size)
        num_warps = 4
    else:
        BT = 64
        BS = 32
        BK = triton.next_power_of_2(K)
        BV = min(triton.next_power_of_2(V), 64)
        num_warps = 2

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)

    if sliding_window_size is not None:
        assert sliding_window_size >= NAtS_block_size, 'The sliding window size must be greater than the nats block size!'

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dk = torch.zeros(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dv = torch.zeros(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    # since we only compute the partial gradients for those selected as attentions, we will not write to all
    # d_nats_block_types. Hence, this needs to be initialized with 0
    dnats = torch.zeros(B, TNAtS * (triton.cdiv(NAtS_block_size, BT)), HQ,
                                    dtype=k.dtype if HNAtS == H else torch.float,
                                    device=q.device)
    grid = (NV, NT, B * HQ)
    stage = 3 if is_causal else 1

    dg_cumsum, dg_cumsum_k = None, None
    if g_cumsum is not None:
        dg_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        dg_cumsum_k = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    n_iters_per_block = compute_attn_n_iters_per_block(
        nats_block_indices,
        T, BT, BS, NAtS_block_size, OFFSET_ATTN,
        0,
        sliding_window_size,
    )
    parallel_nats_attn_bwd_kernel_dq[grid](q=q, k=k, v=v,
                                           g_cumsum=g_cumsum, lse=lse, delta=delta,
                                           do=do, dq=dq, dg_cumsum=dg_cumsum,
                                           nats_block_types=nats_block_types,
                                           nats_block_indices=nats_block_indices,
                                           n_iters_per_block=n_iters_per_block,
                                           cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                                           cu_seqlens_nats=cu_seqlens_nats,
                                           dk=dk, dv=dv, dnats=dnats,
                                           scale=scale,
                                           T=T, TNAtS=TNAtS, B=B, HQ=HQ, HKV=H, G=G,
                                           HNAtS=HNAtS,
                                           GNAtS=GNAtS,
                                           K=K,
                                           V=V,
                                           BT=BT,
                                           BS=BS,
                                           BK=BK,
                                           BV=BV,
                                           NAtS_BLOCK_SIZE=NAtS_block_size,
                                           SW_SIZE=sliding_window_size,
                                           STAGE=stage,
                                           COMPUTE_INCOMPLETE_CHUNK_SCORES=compute_incomplete_chunk_scores,
                                           N_TYPES=N_TYPES,
                                           OFFSET_ATTN=OFFSET_ATTN,
                                           num_warps=num_warps,
                                           )
    # n_grids =prepare_nats_block_indices

    BS = triton.cdiv(BT, NAtS_block_size) * BS  # TODO check if this is really beneficial!!!
    BT = min(BT, NAtS_block_size)  # we need to ensure that each BT block only loads one BT

    if sliding_window_size is not None or compute_dnats_for_invalid_blocks_attn:
        NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
        NV = triton.cdiv(V, BV)
        grid_kv = (NV, NT, B * HQ)

        parallel_attn_bwd_kernel_dkdv_within_valid_blocks[grid_kv](
            q=q, k=k, v=v,
            g_cumsum=g_cumsum, lse=lse, delta=delta,
            do=do, dg_cumsum=dg_cumsum,
            nats_block_types=nats_block_types, nats_block_indices=nats_block_indices,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            dk=dk, dv=dv, dnats=dnats,
            scale=scale,
            T=T, TNAtS=TNAtS, B=B, HQ=HQ, HKV=H, G=G,
            HNAtS=HNAtS,
            GNAtS=GNAtS,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            NAtS_BLOCK_SIZE=NAtS_block_size,
            SW_SIZE=sliding_window_size,
            STAGE=stage,
            COMPUTE_INCOMPLETE_CHUNK_SCORES=compute_incomplete_chunk_scores,
            COMPUTE_DNATS_FOR_INVALID_BLCOKS=compute_dnats_for_invalid_blocks_attn,
            N_TYPES=N_TYPES,
            OFFSET_ATTN=OFFSET_ATTN,
            num_warps=num_warps,
        )
    else:
        chunk_indices_attn_nats = prepare_nats_block_indices(n_nats_blocks[..., OFFSET_ATTN], NAtS_block_size, BT)

        grid_kv = (NV, len(chunk_indices_attn_nats), GNAtS)

        parallel_attn_bwd_kernel_dkdv[grid_kv](
            q=q, k=k, v=v,
            g_cumsum=g_cumsum, lse=lse, delta=delta,
            do=do, dg_cumsum=dg_cumsum,
            nats_block_types=nats_block_types, nats_block_indices=nats_block_indices,
            chunk_indices_attn_nats=chunk_indices_attn_nats,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            cu_seqlens_nats=cu_seqlens_nats,
            dk=dk, dv=dv, dnats=dnats,
            scale=scale,
            T=T, TNAtS=TNAtS, B=B, HQ=HQ, HKV=H, G=G,
            HNAtS=HNAtS,
            GNAtS=GNAtS,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            NAtS_BLOCK_SIZE=NAtS_block_size,
            SW_SIZE=sliding_window_size,
            STAGE=stage,
            COMPUTE_INCOMPLETE_CHUNK_SCORES=compute_incomplete_chunk_scores,
            N_TYPES=N_TYPES,
            OFFSET_ATTN=OFFSET_ATTN,
            num_warps=num_warps,
        )

    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    if NAtS_block_size > BT:
        dnats = reduce(
            dnats, 'b (t gt) (h g) -> b t h',
            g=GNAtS, gt=triton.cdiv(NAtS_block_size, BT), reduction='sum')

    else:
        dnats = reduce(dnats, 'b t (h g) -> b t h', g=GNAtS, reduction='sum')

    if g_cumsum is not None:
        dg_cumsum.add_(dg_cumsum_k)
    return dq, dk, dv, dg_cumsum, dnats


@torch.compile
class AttentionNAtSFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g,
                nats_block_types: torch.Tensor,
                nats_block_indices: torch.LongTensor,
                n_nats_blocks: torch.Tensor,
                scale, cu_seqlens, cu_seqlens_nats,
                NAtS_block_size: int,
                sliding_window_size: int | None = None,
                offset_attn: int = 0,
                compute_incomplete_chunk_scores: bool = True,
                compute_dnats_for_invalid_blocks_attn: bool = False,
                is_causal: bool = True,
                store_msk: bool = False
                ):
        ctx.dtype = q.dtype
        RCP_LN2: float = 1.4426950216

        g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2) if g is not None else None
        o, lse, msk = parallel_attn_nats_fwd(
            q, k, v, nats_block_types, nats_block_indices,
            g_cumsum, scale, NAtS_block_size=NAtS_block_size, sliding_window_size=sliding_window_size,
            offset_attn=offset_attn, compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
            is_causal=is_causal, store_msk=store_msk,
        )
        ctx.save_for_backward(q, k, v, nats_block_types, nats_block_indices, n_nats_blocks, o, g_cumsum, lse)

        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.NAtS_block_size = NAtS_block_size
        ctx.sliding_window_size=sliding_window_size
        ctx.cu_seqlens_nats = cu_seqlens_nats
        ctx.offset_attn = offset_attn
        ctx.compute_incomplete_chunk_scores = compute_incomplete_chunk_scores
        ctx.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        ctx.is_causal = is_causal
        return o.to(q.dtype), msk

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dmsk):
        q, k, v, nats_block_types, nats_block_indices, n_nats_blocks, o, g_cumsum, lse = ctx.saved_tensors
        dq, dk, dv, dg, d_chunk_type = parallel_attn_nats_bwd(
            q, k, v, o, nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            g_cumsum=g_cumsum,
            lse=lse,
            do=do,
            scale=ctx.scale,
            NAtS_block_size=ctx.NAtS_block_size,
            OFFSET_ATTN=ctx.offset_attn,
            compute_incomplete_chunk_scores=ctx.compute_incomplete_chunk_scores,
            compute_dnats_for_invalid_blocks_attn=ctx.compute_dnats_for_invalid_blocks_attn,
            sliding_window_size=ctx.sliding_window_size,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_nats=ctx.cu_seqlens_nats,
            is_causal=ctx.is_causal,
        )
        if nats_block_types.shape[-1] > 1:
            d_chunk_type = torch.nn.functional.pad(d_chunk_type.unsqueeze(-1), (0, nats_block_types.shape[-1] - 1))
        else:
            d_chunk_type = d_chunk_type.unsqueeze(-1)
        return dq.to(q), dk.to(k), dv.to(v), dg, d_chunk_type.to(
            nats_block_types), None, None, None, None, None, None, None, None, None, None, None, None


def parallel_nats_attn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        g,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.LongTensor,
        n_nats_blocks: torch.Tensor,
        scale, cu_seqlens=None, cu_seqlens_nats=None,
        NAtS_block_size: int = 64,
        sliding_window_size: int| None = None,
        offset_attn: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        compute_dnats_for_invalid_blocks_attn: bool=False,
        head_first: bool = False,
        is_causal: bool = True,
        store_msk: bool = False,
):
    """
    This function is implemented for computing the partial attention values. For exactly, we only load the kv values
    where nats_block_types == Attention to compute the attention output.
    the nats_block_types could be determined by chunks, i.e., we determine the types for the tokens for every
    NAtS_block_size based on the token values within the chunk. Hence, we use nats_block_indices to mark the indices
    that are assigned to attention computations. These indices are sorted to allow the triton kernel to extract the
    corresponding columns in the attention maps.
    However, we cannot determine the token types if we only have incomplete chunks as we will only know that chunk type
    after we have observed the last chunk. compute_incomplete_chunk_scores determines if we would like to
    consider these temporary tokens as part of self-attention chunks or not.
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor | None):
            log decay factors of shape `[B, T, H]`.
        nats_block_types (torch.Tensor):
            types of each chunks of shape `[B, TNAtS, H_NAtS, N_opts]`. The types are encoded as one-hot encoding.
            Since we aggregate multiple tokens into one chunk, TNAtS = T / NAtS_block_size. We could also deteremine
            if all the kv heads share the same chunk type or different ones.
        nats_block_indices: (torch.Tensor):
            indices of each chunk, of shape `[B, TNAtS, H_NAtS, N_opts]`, this makes it easier for us to extract the
            required chunks from each column
        n_nats_blocks: (torch.Tensor):
            number of nats chunks of shape `[B, H_NAtS, N_opts]`, numer of nats chunks for each options
        scale (float | None):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        cu_seqlens_nats (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training with nats chunks
            consistent with the FlashAttention API.
        NAtS_block_size (int):
            nats chunk size, how many tokens are merged together to detemine their token types,
        offset_attn (int):
            a value indicates which nats_block_indices is considered as
        compute_incomplete_chunk_scores (bool):
            if we want to consider the incomplete chunk as attention scores
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.
        is_causal (bool):
            if we need to apply a causal mask to the self attention computation
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"

    o, msk = AttentionNAtSFunction.apply(q, k, v, g,
                                         nats_block_types, nats_block_indices, n_nats_blocks,
                                         scale, cu_seqlens,
                                         cu_seqlens_nats, NAtS_block_size, sliding_window_size,
                                         offset_attn,
                                         compute_incomplete_chunk_scores,
                                         compute_dnats_for_invalid_blocks_attn,
                                         is_causal, store_msk
                                         )
    return o, msk


def test_mixed_attns():
    # TODO move this to tests!
    torch.manual_seed(0)
    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16
    #dtype = torch.float16

    import math
    import copy
    BATCH = 4
    HQ = 4
    H = 2
    HAttns = 2
    T = 4135
    #T = 573
    D_HEAD = 64
    GATTN = 1
    N_OPTs = 3
    NATS_Chunk = 64
    TNAtS = triton.cdiv(T, NATS_Chunk)
    BLOCK_M = 32
    device = torch.device("cuda")

    q = torch.randn((BATCH, T, HQ, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, T, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, T, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)

    scale = 1 / math.sqrt(D_HEAD)
    from torch.nn import Transformer

    logits = torch.randn(BATCH, TNAtS, HAttns, N_OPTs, device=device, dtype=dtype)
    attn_types = torch.nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
    # attn_types[...,0] =1.
    attn_idx = torch.where(attn_types == 1., torch.arange(TNAtS, device=attn_types.device).view(1, -1, 1, 1), TNAtS)
    attn_idx = attn_idx.sort(1)[0]
    # We always sum towards the lenght dim
    n_nats_blocks = attn_types.int().sum(1)

    q0 = copy.deepcopy(q.detach())
    k0 = copy.deepcopy(k.detach())
    v0 = copy.deepcopy(v.detach())

    q0 = torch.nn.Parameter(q0, requires_grad=True)
    k0 = torch.nn.Parameter(k0, requires_grad=True)
    v0 = torch.nn.Parameter(v0, requires_grad=True)
    attn_types_ = torch.nn.Parameter(attn_types, requires_grad=True)

    compute_incomplete_chunk_scores = True
    sliding_window_size=None

    out1, msk = parallel_nats_attn(q0.view(BATCH, T, HQ * GATTN, D_HEAD // GATTN),
                                   k0.view(BATCH, T, H * GATTN, D_HEAD // GATTN),
                                   v0.view(BATCH, T, H * GATTN, D_HEAD // GATTN),
                                   nats_block_types=attn_types_,
                                   nats_block_indices=attn_idx,
                                   n_nats_blocks=n_nats_blocks,
                                   offset_attn=0,
                                   scale=scale, cu_seqlens=None, cu_seqlens_nats=None, g=None,
                                   # compute_incomplete_chunk_scores=True,
                                   compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
                                   NAtS_block_size=NATS_Chunk,
                                   store_msk=True
                                   )
    target = torch.rand_like(out1)
    loss = ((out1 - target) ** 2).sum()
    #loss = (out1 ** 2).sum()
    #loss.backward()

    q1 = copy.deepcopy(q.detach()).view(BATCH, T, HQ * GATTN, D_HEAD // GATTN).transpose(1, 2)
    k1 = copy.deepcopy(k.detach()).view(BATCH, T, H * GATTN, D_HEAD // GATTN).transpose(1, 2)
    v1 = copy.deepcopy(v.detach()).view(BATCH, T, H * GATTN, D_HEAD // GATTN).transpose(1, 2)

    q1 = torch.nn.Parameter(q1, requires_grad=True)
    k1 = torch.nn.Parameter(k1, requires_grad=True)
    v1 = torch.nn.Parameter(v1, requires_grad=True)

    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        bs, n_kv_heads, slen, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]
                .expand(bs, n_kv_heads, n_rep, slen, head_dim)
                .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
        )

    def repeat_masks(mask: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This function is similar to repeat_kv, however, we will expand the second dimension
        """
        bs, n_kv_heads, slen1, slen2 = mask.shape
        if n_rep == 1 or n_kv_heads == 1:
            return mask
        return (
            mask[:, :, None, :, :]
                .expand(bs, n_kv_heads, n_rep, slen1, slen2)
                .reshape(bs, n_kv_heads * n_rep, slen1, slen2)
        )

    """
    # This is the case for compute_incomplete_chunk:
    mask = Transformer.generate_square_subsequent_mask(T, device=device, dtype=dtype)

    mask_ = torch.where(mask == 0, 1., 0.).unsqueeze(0).unsqueeze(0)
    valid_columns = attn_types[..., 0].repeat_interleave(NATS_Chunk, dim=1)
    valid_columns = torch.transpose(valid_columns, 1,2).unsqueeze(-1)
    

    mask_ = mask_ * valid_columns
    """
    valid_columns = torch.transpose(attn_types[..., 0], 1, 2)
    full_chunks = (torch.arange(T)[:, None] >= (torch.arange(1, TNAtS + 1)[None, :] * NATS_Chunk)).cuda()

    mask_ = valid_columns.unsqueeze(-2) * full_chunks[None, None, :, :].to(valid_columns)

    mask_ = mask_.repeat_interleave(NATS_Chunk, 3)
    if sliding_window_size is not None:
        mask_sw =  (torch.arange(T)[:, None] >= (torch.arange(T)[None, :])).cuda()
        #mask_sw = mask_sw & (torch.arange(T)[None,:]+sliding_window_size > torch.arange(T)[:, None]).cuda()
        mask_sw = mask_sw & (torch.arange(T)[:, None]-sliding_window_size < (torch.arange(T)[None, :])).cuda()
        mask_ = mask_[:,:,:,:T] + mask_sw.view(1, 1, T, T).to(mask_)
    elif compute_incomplete_chunk_scores:
        # mask_diag =
        mask_diag = (torch.arange(0, NATS_Chunk, device=device)[:, None] >= torch.arange(0, NATS_Chunk, device=device))
        mask_diag = torch.block_diag(*[mask_diag for _ in range(TNAtS)])[:T,:T]
        mask_ = mask_[...,:T] + mask_diag.view(1, 1, T, T).to(mask_)

    mask_ = repeat_masks(mask_[...,:T], HQ // HAttns).bool().to(msk)

    k2 = repeat_kv(k1, HQ // H)
    v2 = repeat_kv(v1, HQ // H)
    out2 = torch.nn.functional.scaled_dot_product_attention(q1, k2, v2, mask_.bool())
    loss2 = ((target-out2.transpose(1, 2).view(BATCH, T, HQ, D_HEAD)) ** 2).sum()
    #loss2 = (out2**2).sum()
    #loss2.backward()

    out2 = out2.transpose(1, 2).view(BATCH, T, HQ, D_HEAD)

    print(f'out diff max:{(out2 - out1).max()}')
    print(f'out diff min:{(out2 - out1).min()}')

    import pdb
    pdb.set_trace()
    print(f'q grad max: {(q0.grad - q1.grad.transpose(1, 2)).max()}')
    print(f'q grad min: {(q0.grad - q1.grad.transpose(1, 2)).min()}')

    print(f'k grad max: {(k0.grad - k1.grad.transpose(1, 2)).max()}')
    print(f'k grad max: {(k0.grad - k1.grad.transpose(1, 2)).min()}')

    print(f'v grad max: {(v0.grad - v1.grad.transpose(1, 2)).max()}')
    print(f'v grad max: {(v0.grad - v1.grad.transpose(1, 2)).min()}')

    import pdb
    pdb.set_trace()

    for b in range(BATCH):
        for h in range(H):
            print(f"batch: {b}, head:{h}")

            k_grad_idx = torch.where(k0.grad[b, :, h, 0])[0]
            v_grad_idx = torch.where(v0.grad[b, :, h, 0])[0]
            print(f'kv grad idx diff {(k_grad_idx - v_grad_idx).abs().sum()}')
            attn_type_grad_idx = torch.where(attn_types_.grad[b, :, h // (H // HAttns), 0] != 0)[0]
            attn_type_grad_idx = attn_type_grad_idx[:, None] * NATS_Chunk + torch.arange(NATS_Chunk).cuda()[None, :]
            attn_type_grad_idx = attn_type_grad_idx.flatten()
            import pdb
            pdb.set_trace()
            print(f'k, chunk grad idx diff {(k_grad_idx - attn_type_grad_idx).abs().sum()}')
            print('\n')

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    test_mixed_attns()
    # bench_flash_attention(BATCH, N_HEADS, N_CTX, D_HEAD, True, 'bwd', 'triton', False)
    # bench_flash_attention.run(save_path=".", print_data=True)
# """
