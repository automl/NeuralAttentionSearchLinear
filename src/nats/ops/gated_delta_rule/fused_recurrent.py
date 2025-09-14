# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.utils import input_guard


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.jit(do_not_specialize=['T', 'TNAtS'])
def fused_recurrent_gated_delta_rule_fwd_nats_kernel(
    q,
    k,
    v,
    g,
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
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    i_hnats = i_h // GNAtS
    i_gnats = i_h % GNAtS

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos

        bos_nats, eos_nats = tl.load(cu_seqlens_nats + i_b).to(tl.int32), tl.load(cu_seqlens_nats + i_b + 1).to(
            tl.int32)
        TNAtS = eos_nats - bos_nats
    else:
        bos, eos = i_n * T, i_n * T + T

        bos_nats, eos_nats = i_n * TNAtS, i_n * TNAtS + TNAtS

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

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

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_hcur = hcur + i_nh * K * V + o_k[:, None] * V + o_v[None, :]

    nats_block_types += (bos_nats * HNAtS + i_hnats) * N_TYPES + OFFSET_OP

    b_ntokens_in_current_block = tl.load(n_tokens_in_current_block + i_n)

    n_iter1 = min(T, NATS_BLOCK_SIZE - b_ntokens_in_current_block)

    # the first iteration
    for _ in range(0, n_iter1):
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

    # TODO we need to check how to optimize this?
    if n_iter1 < NATS_BLOCK_SIZE - b_ntokens_in_current_block: # TODO take this as constexpr?
        tl.store(p_hcur, b_h.to(p_hcur.dtype.element_ty), mask=mask_h)
    else:
        is_delta = tl.load(nats_block_types)
        if is_delta:
            tl.store(p_hcur, b_h.to(p_hcur.dtype.element_ty), mask=mask_h)
        else:
            b_h = tl.load(p_hcur)

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
        if is_delta:
            tl.store(p_hcur, b_h.to(p_hcur.dtype.element_ty), mask=mask_h)
        else:
            b_h = tl.load(p_hcur)

    # the last block
    n_iters_last = T - b_ntokens_in_current_block - (TNAtS - 2) * NATS_BLOCK_SIZE
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
    if n_iters_last == NATS_BLOCK_SIZE and is_delta:
        tl.store(p_hcur, b_h.to(p_hcur.dtype.element_ty), mask=mask_h)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_nats_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    nats_block_types: Optional[torch.Tensor] = None,
    n_tokens_in_current_block: Optional[torch.Tensor] = 0,
    nats_block_size: int=64,
    scale: float = None,
    offset_op: int = 1,
    initial_state: torch.Tensor = None,
    initial_state_in_current_block: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_nats: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    B, TNAtS, HNAtS, N_TYPES = nats_block_types.shape
    HV = v.shape[2]
    GNAtS = H // HNAtS
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NV = triton.cdiv(V, BV)
    num_stages = 3
    num_warps = 1

    o = torch.empty_like(v)
    final_state = q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    if initial_state_in_current_block is None:
        if initial_state is not None:
            initial_state_in_current_block = initial_state.clone()
        else:
            initial_state_in_current_block = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_nats_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
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
        N_TYPES=N_TYPES,
        NATS_BLOCK_SIZE=nats_block_size,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        OFFSET_OP=offset_op,
        num_warps=num_warps,
        num_stages=num_stages,
    )
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
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_nats_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
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
    gk: Optional[torch.Tensor] = None,
    gv: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    nats_block_types: Optional[torch.Tensor] = None,
    n_tokens_in_current_block: Optional[torch.Tensor] = 0,
    nats_block_size:int = 64,
    scale: float = None,
    offset_op: Optional[torch.Tensor] = 1,
    initial_state: torch.Tensor = None,
    initial_state_current: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
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
        cu_seqlens,
        cu_seqlens_nats,
    )
    return o, final_state, initial_state_in_current_block
