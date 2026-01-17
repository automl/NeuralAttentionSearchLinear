# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import copy
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import input_guard

@triton.jit
def exp(x):
    return tl.exp(x)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'HAS_G_CUMSUM': lambda args: args['g_cumsum'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T', 'TNAtS', 'STRIDE_KB', 'STRIDE_VB'])
def fused_recurrent_gated_delta_rule_fwd_nats_kernel(
    q,
    k,
    v,
    g,
    g_cumsum,
    gk,
    gv,
    beta,
    o,
    nats_block_types,
    n_tokens_in_current_block,
    h0,
    ht,
    hcur,
    cu_seqlens,
    cu_seqlens_nats,
    scale,
    T,
    TNAtS,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    HNAtS: tl.constexpr,
    GNAtS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_TYPES: tl.constexpr,
    OFFSET_OP: tl.constexpr,
    NATS_BLOCK_SIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STRIDE_KB,
    STRIDE_VB,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    ONLY_UPDATE_H: tl.constexpr,
    ONLY_WITH_CURRENT_CHUNK: tl.constexpr,
    UPDATE_HS_FOR_EACH_ITER: tl.constexpr,
    DECAY_FOR_NON_GDN_BLOCKS: tl.constexpr,
    ONLY_UPDATE_HIDDEN_STATES: tl.constexpr,
    HAS_G_CUMSUM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    i_hnats = i_h // GNAtS
    i_gnats = i_h % GNAtS

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_b).to(tl.int32), tl.load(cu_seqlens_nats + i_b + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
    else:
        bos, eos = i_n * T, i_n * T + T

        bos_nats, eos_nats = i_n * TNAtS, i_n * TNAtS + TNAtS

        p_k = k + i_n * STRIDE_KB + i_h * K + o_k
        p_v = v + i_n * STRIDE_VB + i_h * V + o_v
    if not ONLY_UPDATE_HIDDEN_STATES:
        p_q = q + (bos * H + i_h) * K + o_k
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    if HAS_G_CUMSUM:
        p_g_cumsum = g_cumsum + i_n * HV + i_hv
        b_g_cumsum = tl.load(p_g_cumsum)
    else:
        b_g_cumsum = 0.
    if not ONLY_UPDATE_HIDDEN_STATES:
        p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if not ONLY_UPDATE_HIDDEN_STATES:
        if USE_INITIAL_STATE:
            p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_hcur = hcur + i_nh * K * V + o_k[:, None] * V + o_v[None, :]

    b_hcur = tl.load(p_hcur, mask=mask_h, other=0).to(tl.float32)
    if ONLY_UPDATE_HIDDEN_STATES:
        b_h = b_hcur

    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    b_ntokens_in_current_block = tl.load(n_tokens_in_current_block + i_n)

    n_iter1 = min(T, NATS_BLOCK_SIZE - b_ntokens_in_current_block)

    # the first iteration
    if ONLY_UPDATE_HIDDEN_STATES:
        is_delta = tl.load(nats_block_types).to(tl.int1)
        if not is_delta:
            if USE_G:
                if DECAY_FOR_NON_GDN_BLOCKS:
                    # we only need to decay the input g
                    b_g = tl.load(p_g + tl.arange(0, NATS_BLOCK_SIZE) * HV, mask=tl.arange(0, NATS_BLOCK_SIZE)<T, other=0)
                    b_hcur = b_hcur * exp(tl.sum(b_g))
                    tl.store(p_hcur, b_hcur.to(p_hcur.dtype.element_ty), mask=mask_h)
            return
    for _ in range(0, n_iter1):
        if not ONLY_UPDATE_HIDDEN_STATES:
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)

        if UPDATE_HS_FOR_EACH_ITER:
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            if not ONLY_UPDATE_HIDDEN_STATES:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            if UPDATE_HS_FOR_EACH_ITER:
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        if not ONLY_UPDATE_HIDDEN_STATES:
            b_q = b_q * scale

        if UPDATE_HS_FOR_EACH_ITER:
            if IS_BETA_HEADWISE:
                b_beta = tl.load(p_beta).to(tl.float32)
            else:
                b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        # [BK, BV]
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
            if DECAY_FOR_NON_GDN_BLOCKS:
                b_g_cumsum += b_g

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            b_h *= exp(b_gv[None, :])
        if UPDATE_HS_FOR_EACH_ITER:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v

        # [BV]
        if not ONLY_UPDATE_HIDDEN_STATES:
            b_o = tl.sum(b_h * b_q[:, None], 0)

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        if not UPDATE_HS_FOR_EACH_ITER:
            p_o += HV*V
    if not ONLY_UPDATE_H:
        if not ONLY_WITH_CURRENT_CHUNK:

            # TODO we need to check how to optimize this?
            # we arrive the end of the first block

            if n_iter1 == NATS_BLOCK_SIZE - b_ntokens_in_current_block:
                is_delta = tl.load(nats_block_types)
                if DECAY_FOR_NON_GDN_BLOCKS:
                    b_h = tl.where(is_delta, b_h, b_hcur * tl.exp(b_g_cumsum))
                else:
                    b_h = tl.where(is_delta, b_h, b_hcur)

                b_hcur = b_h
            # Now we want to continue the following
            for _ in range(0, TNAtS - 2):
                for _ in range(0, NATS_BLOCK_SIZE):
                    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                    if USE_QK_L2NORM_IN_KERNEL:
                        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                    b_q = b_q * scale
                    if IS_BETA_HEADWISE:
                        b_beta = tl.load(p_beta).to(tl.float32)
                    else:
                        b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

                    # [BK, BV]
                    if USE_G:
                        b_g = tl.load(p_g).to(tl.float32)
                        b_h *= exp(b_g)

                    if USE_GK:
                        b_gk = tl.load(p_gk).to(tl.float32)
                        b_h *= exp(b_gk[:, None])

                    if USE_GV:
                        b_gv = tl.load(p_gv).to(tl.float32)
                        b_h *= exp(b_gv[None, :])

                    b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
                    b_h += b_k[:, None] * b_v

                    # [BV]
                    b_o = tl.sum(b_h * b_q[:, None], 0)
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

                    p_q += H * K
                    p_k += H * K
                    p_v += HV * V
                    if USE_G:
                        p_g += HV
                    if USE_GK:
                        p_gk += HV * K
                    if USE_GV:
                        p_gv += HV * V
                    p_beta += HV * (1 if IS_BETA_HEADWISE else V)
                    p_o += HV * V
                nats_block_types += N_TYPES * HNAtS
                is_delta = tl.load(nats_block_types)
                if DECAY_FOR_NON_GDN_BLOCKS:
                    b_h = tl.where(is_delta, b_h, b_hcur * tl.exp(b_g_cumsum))
                else:
                    b_h = tl.where(is_delta, b_h, b_hcur)
                b_hcur = b_h

            # the last block
            n_iters_last = T - n_iter1 - max(TNAtS - 2, 0) * NATS_BLOCK_SIZE

            for _ in range(0, n_iters_last):
                b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
                if USE_QK_L2NORM_IN_KERNEL:
                    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                b_q = b_q * scale
                if IS_BETA_HEADWISE:
                    b_beta = tl.load(p_beta).to(tl.float32)
                else:
                    b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

                # [BK, BV]
                if USE_G:
                    b_g = tl.load(p_g).to(tl.float32)
                    b_h *= exp(b_g)

                if USE_GK:
                    b_gk = tl.load(p_gk).to(tl.float32)
                    b_h *= exp(b_gk[:, None])

                if USE_GV:
                    b_gv = tl.load(p_gv).to(tl.float32)
                    b_h *= exp(b_gv[None, :])

                b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
                b_h += b_k[:, None] * b_v

                # [BV]
                b_o = tl.sum(b_h * b_q[:, None], 0)
                tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

                p_q += H*K
                p_k += H*K
                p_v += HV*V
                if USE_G:
                    p_g += HV
                if USE_GK:
                    p_gk += HV*K
                if USE_GV:
                    p_gv += HV*V
                p_beta += HV * (1 if IS_BETA_HEADWISE else V)
                p_o += HV*V

            # we only store the temporal hidden state if we arrive the new hidden state and the new block is a delta block
            nats_block_types += N_TYPES * HNAtS
            is_delta = tl.load(nats_block_types)

            if n_iters_last == NATS_BLOCK_SIZE:
                if DECAY_FOR_NON_GDN_BLOCKS:
                    b_h = tl.where(is_delta, b_h, b_hcur * tl.exp(b_g_cumsum))
                else:
                    b_h = tl.where(is_delta, b_h, b_hcur)
                b_hcur = b_h
        else:
            if ONLY_UPDATE_HIDDEN_STATES:
                b_hcur = b_h
            else:
                is_delta = tl.load(nats_block_types)
                if DECAY_FOR_NON_GDN_BLOCKS:
                    b_h = tl.where(is_delta, b_h, b_hcur * tl.exp(b_g_cumsum))
                else:
                    b_h = tl.where(is_delta, b_h, b_hcur)
                b_hcur = b_h

        # we need to store b_hcur anyway
        tl.store(p_hcur, b_hcur.to(p_hcur.dtype.element_ty), mask=mask_h)

    if STORE_FINAL_STATE:
        if not ONLY_UPDATE_HIDDEN_STATES:
            p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_nats_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor]=None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    nats_block_types: Optional[torch.Tensor] = None,
    n_tokens_in_current_block: Optional[torch.Tensor | int] = 0,
    nats_block_size: int=64,
    scale: float = None,
    offset_op: int = 1,
    initial_state: torch.Tensor = None,
    initial_state_in_current_block: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    update_hs_for_each_iter: bool = False,
    decay_for_non_gdn_blocks: bool= False,
    only_update_hidden_states:bool=False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_nats: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not only_update_hidden_states:
        B, T, H, K, V = *q.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    B, TNAtS, HNAtS, N_TYPES = nats_block_types.shape
    HV = v.shape[2]
    GNAtS = H // HNAtS
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NV = triton.cdiv(V, BV)
    num_stages = 3
    num_warps = 1
    if not only_update_hidden_states:
        o = torch.empty(B, T, H, V, device=q.device, dtype=q.dtype)
    else:
        o = None
    if not only_update_hidden_states and output_final_state:
        final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None
    if initial_state_in_current_block is None:
        if initial_state is not None:
            initial_state_in_current_block = initial_state.clone()
        else:
            initial_state_in_current_block = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    if isinstance(n_tokens_in_current_block, int):
        only_update_h = n_tokens_in_current_block + T < nats_block_size
        n_tokens_in_current_block = torch.full(
            [N], fill_value=n_tokens_in_current_block, device=v.device, dtype=torch.int32
        )
    else:
        n_new_tokens = torch.max(n_tokens_in_current_block) + T
        only_update_h = n_new_tokens < nats_block_size
    only_with_current_chunk = TNAtS == 1

    if only_update_hidden_states:
        update_hs_for_each_iter=True

    if update_hs_for_each_iter:
        assert only_with_current_chunk, "if we do not update hs for each iteration, we can only compute output wihtin " \
                                        "the current iteration"
    assert len(n_tokens_in_current_block) == B

    grid = (NV, N * HV)
    if only_update_hidden_states:
        initial_state_in_current_block1 = initial_state_in_current_block.clone()

    fused_recurrent_gated_delta_rule_fwd_nats_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_cumsum=g_cumsum,
        gk=gk,
        gv=gv,
        beta=beta,
        o=o,
        nats_block_types=nats_block_types,
        n_tokens_in_current_block=n_tokens_in_current_block,
        h0=initial_state,
        ht=final_state,
        hcur=initial_state_in_current_block,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        scale=scale,
        T=T,
        TNAtS=TNAtS,
        B=B,
        H=H,
        HV=HV,
        HNAtS=HNAtS,
        GNAtS=GNAtS,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        STRIDE_KB=k.stride(0),
        STRIDE_VB=v.stride(0),
        N_TYPES=N_TYPES,
        NATS_BLOCK_SIZE=nats_block_size,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        OFFSET_OP=offset_op,
        ONLY_UPDATE_H=only_update_h,
        ONLY_WITH_CURRENT_CHUNK=only_with_current_chunk,
        UPDATE_HS_FOR_EACH_ITER=update_hs_for_each_iter,
        DECAY_FOR_NON_GDN_BLOCKS=decay_for_non_gdn_blocks,
        ONLY_UPDATE_HIDDEN_STATES=only_update_hidden_states,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if only_update_hidden_states:
        import pdb
        pdb.set_trace()

    return o, final_state, initial_state_in_current_block


class FusedRecurrentNAtSFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        g_cumsum: Optional[torch.Tensor] = None,
        gk: Optional[torch.Tensor] = None,
        gv: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        nats_block_types: Optional[torch.Tensor] = None,
        n_tokens_in_current_block: Optional[torch.Tensor] = 0,
        nats_block_size: int = 64,
        scale: float = None,
        offset_op: int = 1,
        initial_state: torch.Tensor = None,
        initial_state_in_current_block: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        update_hs_for_each_iter: bool= True,
        decay_for_non_gdn_blocks: bool= False,
        only_update_hidden_states: bool=False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
    ):
        o, final_state, initial_state_in_current_block = fused_recurrent_gated_delta_rule_nats_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=g_cumsum,
            gk=gk,
            gv=gv,
            beta=beta,
            nats_block_types=nats_block_types,
            n_tokens_in_current_block=n_tokens_in_current_block,
            nats_block_size=nats_block_size,
            scale=scale,
            initial_state=initial_state,
            initial_state_in_current_block=initial_state_in_current_block,
            offset_op=offset_op,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            update_hs_for_each_iter=update_hs_for_each_iter,
            decay_for_non_gdn_blocks=decay_for_non_gdn_blocks,
            only_update_hidden_states=only_update_hidden_states,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
        )

        return o, final_state, initial_state_in_current_block

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps."
        )


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    nats_block_types: Optional[torch.Tensor] = None,
    n_tokens_in_current_block: torch.Tensor | int = 0,
    nats_block_size:int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_current: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    offset_op: Optional[int] = 1,
    update_hs_for_each_iter: bool=False,
    decay_for_non_gdn_blocks: bool=False,
    only_update_hidden_states: bool=False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_nats:Optional[torch.LongTensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    o, final_state, initial_state_in_current_block = FusedRecurrentNAtSFunction.apply(
        q,
        k,
        v,
        g,
        g_cumsum,
        gk,
        gv,
        beta,
        nats_block_types,
        n_tokens_in_current_block,
        nats_block_size,
        scale,
        offset_op,
        initial_state,
        initial_state_current,
        output_final_state,
        use_qk_l2norm_in_kernel,
        update_hs_for_each_iter,
        decay_for_non_gdn_blocks,
        only_update_hidden_states,
        cu_seqlens,
        cu_seqlens_nats,
    )
    return o, final_state, initial_state_in_current_block


def test_recurrent_fwd():
    torch.manual_seed(0)
    from nats.utils import check_fp16_dtype
    import triton
    from torch.nn import functional as F
    import math
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16
    dtype = torch.float16

    n_data_in_ctx = 63

    B = 2
    H = 4
    HNatS = 4
    T = 1
    N_TYPES = 3
    GNAtS = H // HNatS
    delta_offset = 1
    K = 128
    V = 256
    NATS_Chunk = 64
    DELTA_IDX = 1

    T_NAtS = triton.cdiv(T + n_data_in_ctx, NATS_Chunk)

    device = torch.device('cuda')
    q = torch.randn(B, T, H, K, dtype=dtype, device=torch.device('cuda'))
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device='cuda'), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=dtype, device=torch.device('cuda'))
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device) * 20)

    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=torch.device('cuda'))
    h0_chunk_start = torch.randn(B, H, K, V, dtype=torch.float32, device=torch.device('cuda'))

    logits = torch.randn(B, T_NAtS, HNatS, N_TYPES, device=torch.device('cuda'), dtype=dtype)
    nats_block_types = torch.nn.functional.gumbel_softmax(logits, dim=-1, hard=True)

    scale = 1 / math.sqrt(K)
    t_current = 0
    all_output = []
    state_tmp = copy.deepcopy(h0)
    state_chunk_start = copy.deepcopy(h0_chunk_start)
    for i_chunk in range(T_NAtS):
        if i_chunk == 0:
            t_end = NATS_Chunk - n_data_in_ctx
            q_in = q[:, t_current:t_end, ]
            k_in = k[:, t_current:t_end ]
            v_in = v[:, t_current:t_end]
            beta_in = beta[:, t_current:t_end]
            g_in = g[:, t_current:t_end]
            t_current = t_end
        else:
            t_end = min(t_current + NATS_Chunk, T)
            q_in = q[:, t_current:t_end, ]
            k_in = k[:, t_current:t_end ]
            v_in = v[:, t_current:t_end]
            beta_in = beta[:, t_current:t_end]
            g_in = g[:, t_current:t_end]
            t_current = t_end

        o1, final_state1 = fused_recurrent_gated_delta_rule_fwd(
            q=q_in.contiguous(), k=k_in.contiguous(), v=v_in.contiguous(), g=g_in.contiguous(),
            beta=beta_in.contiguous(), scale=scale, initial_state=state_tmp,
            output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        all_output.append(o1)
        #if not ((i_chunk == T_NAtS - 1) and (k_in.shape[1] < NATS_Chunk)):
        if True: # TODO for test purpose, this might need to be further optimized...
            for b in range(B):
                for h in range(H):
                    block_is_delta = nats_block_types[b, i_chunk, h, DELTA_IDX]
                    if block_is_delta:
                        state_chunk_start[b, h] = final_state1[b,h]
                    state_tmp[b,h] = state_chunk_start[b,h]

    o1 = torch.cat(all_output, dim=1)
    final_state_chunkstart1 = state_chunk_start

    o, final_state, initial_state_in_current_block = fused_recurrent_gated_delta_rule_nats_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        gk=None,
        gv=None,
        beta=beta,
        nats_block_types=nats_block_types,
        n_tokens_in_current_block=n_data_in_ctx,
        nats_block_size=NATS_Chunk,
        scale=scale,
        initial_state=copy.deepcopy(h0),
        initial_state_in_current_block=copy.deepcopy(h0_chunk_start),
        offset_op=DELTA_IDX,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        cu_seqlens_nats=None,
    )
    print(f"diff o: {(o-o1).abs().max()}")
    print(f"diff states: {(final_state1-final_state).abs().max()}")
    print(f"diff state_chunk: {(final_state_chunkstart1-initial_state_in_current_block).abs().max()}")


    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    test_recurrent_fwd()

