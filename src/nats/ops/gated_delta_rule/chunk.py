# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from nats.ops.common.chunk_delta_h import chunk_gated_delta_rule_nats_bwd_dhu, chunk_gated_delta_rule_nats_fwd_h
# from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from nats.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_nats_fwd
from nats.ops.gated_delta_rule.wy_fast import prepare_wy_repr_nats_bwd, prepare_wy_repr_nats_fwd, recompute_w_u_nats_fwd
from nats.ops.common.chunk_o import chunk_bwd_nats_dqkwg, chunk_bwd_dv_qdo_nats_local, chunk_fwd_nats_o
from nats.ops.utils.cumsum import chunk_cumsum_non_gated_chunks
from fla.ops.utils import chunk_local_cumsum
from nats.ops.utils import solve_tril_nats
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks


# TODO in the current implementation, we use O_{t} = Q_{t} @ h_{t0} + p_t @ (v_t - w_t @ h_t0)
#  However, an alternative could also be   O_{t} = Q_{t} @ h_{t0} + p_t @ (v_t), which requires a
#  different fwd and bwd implementation...

def chunk_gated_delta_rule_nats_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        chunk_indices_delta_nats: torch.Tensor,
        nats_block_delta_offsets: torch.Tensor,
        starting_h_idx_delta: torch.Tensor,
        decay_for_non_gdn_blocks: bool= True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        nats_block_size: int = 64,
        offset_delta: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        incomplete_block_start_with_ht: bool = True,
        keep_wu_as_kv: bool = False,
        chunk_size: int = 64
):
    assert nats_block_size >= chunk_size, "The current implementaion only allows one nats block within each delta " \
                                          "computaionl chunk!!!"
    # TODO the current implementation does not consider the cases where NAtSChunk Size is greater than BT, we need to
    #  check that in the future implementation !!!

    g = chunk_local_cumsum(g, chunk_size=64,
                           cu_seqlens=cu_seqlens,
                           )
    if decay_for_non_gdn_blocks:
        g = chunk_cumsum_non_gated_chunks(
            g_cumsum=g,
            chunk_size=chunk_size,
            nats_block_size=nats_block_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            chunk_indices_op_nats=chunk_indices_delta_nats,
            reversed=False,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            offset_op=offset_delta,
        )

    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_nats_fwd(
        k=k,
        beta=beta,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        g=g,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        offset_op=offset_delta,
        output_dtype=torch.float32,
        keep_wu_as_kv=keep_wu_as_kv,
    )

    A = solve_tril_nats(
        A=A,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        n_nats_blocks=n_nats_blocks,
        nats_block_indices=nats_block_indices,
        nats_block_size=nats_block_size,
        T=q.shape[1],
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        offset_op=offset_delta,
        output_dtype=k.dtype,
        keep_wu_as_kv=keep_wu_as_kv,
    )
    w, u = recompute_w_u_nats_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        g=g,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        keep_wu_as_kv=keep_wu_as_kv,
    )
    # We note that here only the parts that are gated_delta parts are updated with v
    # For the other part, it we have incomplete_block_start_with_ht, they will be updated within the
    # chunk_fwd_nats_o function,
    h, v_new, final_state = chunk_gated_delta_rule_nats_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=None,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        nats_block_delta_offsets=nats_block_delta_offsets,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        keep_wu_as_kv=keep_wu_as_kv,
        decay_for_non_gdn_blocks=decay_for_non_gdn_blocks,
    )
    o = chunk_fwd_nats_o(
        q=q,
        k=k,
        v=v_new if incomplete_block_start_with_ht else u,
        w=w,
        h=h,
        g=g,
        g_gamma=None,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        nats_block_op_offsets=nats_block_delta_offsets,
        starting_h_idx=starting_h_idx_delta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        offset_op=offset_delta,
        compute_incomplete_block_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        decay_for_non_gdn_blocks=decay_for_non_gdn_blocks,
        keep_wu_as_kv=keep_wu_as_kv,
    )
    return g, o, A, final_state


def chunk_gated_delta_rule_nats_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        nats_block_types: torch.Tensor,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        chunk_indices_delta_nats: torch.Tensor,
        nats_block_delta_offsets: torch.Tensor,
        starting_h_idx_delta: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        nats_block_size: int = 1,
        offset_delta: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        compute_dnats_for_invalid_blocks: bool = True,
        incomplete_block_start_with_ht: bool = True,
        keep_wu_as_kv: bool = True,
        decay_for_non_gdn_blocks:bool=False,
        chunk_size: int = 64,
):
    assert keep_wu_as_kv == (compute_incomplete_chunk_scores or compute_dnats_for_invalid_blocks)
    w, u = recompute_w_u_nats_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        g=g,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        keep_wu_as_kv=keep_wu_as_kv,
    )
    h, v_new, _ = chunk_gated_delta_rule_nats_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=None,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        nats_block_delta_offsets=nats_block_delta_offsets,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        keep_wu_as_kv=keep_wu_as_kv,
        decay_for_non_gdn_blocks=decay_for_non_gdn_blocks,
    )
    dv, qdo, v_new = chunk_bwd_dv_qdo_nats_local(q=q,
                                                 k=k,
                                                 w=w,
                                                 do=do,
                                                 v=v_new,
                                                 h=h,
                                                 nats_block_types=nats_block_types,
                                                 nats_block_indices=nats_block_indices,
                                                 n_nats_blocks=n_nats_blocks,
                                                 chunk_indices_op_nats=chunk_indices_delta_nats,
                                                 g=g,
                                                 scale=scale,
                                                 nats_block_size=nats_block_size,
                                                 offset_op=offset_delta,
                                                 compute_incomplete_block_scores=compute_incomplete_chunk_scores,
                                                 incomplete_block_start_with_ht=incomplete_block_start_with_ht,
                                                 decay_for_non_gdn_blocks=decay_for_non_gdn_blocks,
                                                 )

    dh, dh0, dv = chunk_gated_delta_rule_nats_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        qdo=qdo,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        nats_block_delta_offsets=nats_block_delta_offsets,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        keep_wu_as_kv=keep_wu_as_kv,
        decay_for_non_gdn_blocks=decay_for_non_gdn_blocks
    )

    dq, dk, dw, dg, d_nats = chunk_bwd_nats_dqkwg(
        q=q, k=k, v=v_new, do=do, h=h, dh=dh, nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        nats_block_op_offsets=nats_block_delta_offsets,
        n_nats_blocks=n_nats_blocks,
        starting_h_idx=starting_h_idx_delta,
        chunk_indices_op_nats=chunk_indices_delta_nats, g=g, g_gamma=None,
        dv=dv, w=w, cu_seqlens=cu_seqlens, cu_seqlens_nats=cu_seqlens_nats,
        nats_block_size=nats_block_size, scale=scale, offset_op=offset_delta,
        compute_incomplete_block_scores=compute_incomplete_chunk_scores,
        compute_dnats_for_invalid_blocks=compute_dnats_for_invalid_blocks,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        decay_for_non_gdn_blocks=decay_for_non_gdn_blocks,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_nats_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        keep_wu_as_kv=keep_wu_as_kv,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    if decay_for_non_gdn_blocks:
        dg = chunk_cumsum_non_gated_chunks(
            g_cumsum=g,
            chunk_size=chunk_size,
            nats_block_size=nats_block_size,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            chunk_indices_op_nats=chunk_indices_delta_nats,
            reversed=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            offset_op=offset_delta,
        )
    return dq, dk, dv, db, dg, dh0, d_nats


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            nats_block_types: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            nats_block_indices: torch.Tensor,
            n_nats_blocks: torch.Tensor,
            cu_seqlens: Optional[torch.LongTensor] = None,
            cu_seqlens_nats: Optional[torch.LongTensor] = None,
            nats_block_size: int = 64,
            offset_delta: int = 0,
            compute_incomplete_chunk_scores: bool = False,
            compute_dnats_for_invalid_blocks: bool = True,
            use_qk_l2norm_in_kernel: bool = False,
            chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        chunk_indices_delta_nats = prepare_nats_block_indices(n_nats_blocks[..., offset_delta],
                                                              nats_block_size,
                                                              chunk_size, )
        nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                              nats_block_types,
                                                              nats_block_size,
                                                              chunk_size, offset_delta)
        starting_h_idx = compute_starting_idx_for_chunks(
            nats_chunk_indices=nats_block_indices,
            T=q.shape[1],
            BT=chunk_size,
            NAtS_Block_Size=nats_block_size,
            offset_op=offset_delta
        )

        g, o, A, final_state = chunk_gated_delta_rule_nats_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_indices_delta_nats=chunk_indices_delta_nats,
            nats_block_delta_offsets=nats_block_delta_offsets,
            starting_h_idx_delta=starting_h_idx,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=nats_block_size,
            offset_delta=offset_delta,
            compute_incomplete_chunk_scores=compute_incomplete_chunk_scores
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens, cu_seqlens_nats,
                              nats_block_types, nats_block_indices, n_nats_blocks, chunk_indices_delta_nats,
                              nats_block_delta_offsets, starting_h_idx,
                              )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.nats_block_size = nats_block_size
        ctx.offset_delta = offset_delta
        ctx.compute_incomplete_chunk_scores = compute_incomplete_chunk_scores
        ctx.compute_dnats_for_invalid_blocks = compute_dnats_for_invalid_blocks
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
            ctx,
            do: torch.Tensor,
            dht: torch.Tensor
    ):
        (
            q, q_rstd, k, k_rstd, v, g, beta,
            A, initial_state, cu_seqlens, cu_seqlens_nats, nats_block_types,
            nats_block_indices, n_nats_blocks, chunk_indices_delta_nats,
            nats_block_delta_offsets, starting_h_idx
        ) = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0, d_nats = chunk_gated_delta_rule_nats_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            chunk_indices_delta_nats=chunk_indices_delta_nats,
            nats_block_delta_offsets=nats_block_delta_offsets,
            starting_h_idx_delta=starting_h_idx,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=ctx.nats_block_size,
            offset_delta=ctx.offset_delta,
            compute_incomplete_chunk_scores=ctx.compute_incomplete_chunk_scores,
            compute_dnats_for_invalid_blocks=ctx.compute_dnats_for_invalid_blocks,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        head_first: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        warnings.warn(
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
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel
    )
    return o, final_state


def test_compute_h(dtype=torch.bfloat16):
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
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    g0 = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device) * 20)

    logits = torch.randn(B, T_NAtS, HNatS, N_TYPES, device=torch.device('cuda'), dtype=dtype)
    nats_block_types = torch.nn.functional.gumbel_softmax(logits, dim=-1, hard=True)
    # attN_TYPES[...,0] =1.
    # we ask all the models to have the last idx as valid
    nats_block_types[:, -1, :] = 1.
    nats_block_indices = torch.where(nats_block_types == 1.,
                                     torch.arange(T_NAtS, device=nats_block_types.device).view(1, -1, 1, 1), T_NAtS)
    nats_block_indices = nats_block_indices.sort(1)[0]
    n_nats_blocks = torch.sum(nats_block_types.long(), dim=1)

    import copy
    output_final_state = True
    cu_seqlens = None
    mask_ = nats_block_types[..., delta_offset]
    mask_ = mask_.repeat_interleave(NATS_Chunk, 1)
    mask_ = mask_[:, :, :, None].expand(B, NATS_Chunk * T_NAtS, HNatS, GNAtS).reshape(B, NATS_Chunk * T_NAtS,
                                                                                      HNatS * GNAtS)
    mask_ = mask_[:, :T, :]
    q1 = copy.deepcopy(q)
    k1 = copy.deepcopy(k) * mask_[..., None]
    v1 = copy.deepcopy(v) * mask_[..., None]
    beta1 = copy.deepcopy(beta) * mask_  # should not be necessary here
    # k1 = copy.deepcopy(k)
    # v1 = copy.deepcopy(v)
    # beta1 = copy.deepcopy(beta)
    import math
    scale = 1 / math.sqrt(K)
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.utils import solve_tril, chunk_local_cumsum
    # from fla.ops.common.
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    # """
    # delta net fwd
    g1 = chunk_local_cumsum(copy.deepcopy(g0) * mask_, chunk_size=64, cu_seqlens=None)

    A1 = chunk_scaled_dot_kkt_fwd(
        k=k1,
        g=g1,
        beta=beta1,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )

    A1 = solve_tril(
        A=A1,
        cu_seqlens=cu_seqlens,
        output_dtype=k1.dtype
    )

    w1, u1 = recompute_w_u_fwd(
        k=k1,
        v=v1,
        beta=beta1,
        A=A1,
        g=g1,
        cu_seqlens=cu_seqlens,
    )
    h1, v_new1, final_state1 = chunk_gated_delta_rule_fwd_h(
        k=k1,
        w=w1,
        u=u1,
        g=g1,
        initial_state=None,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        save_new_value=True
    )
    o1 = chunk_fwd_o(
        q=q1,
        k=k1,
        v=v_new1,
        h=h1,
        g=g1,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    # """
    """
    torch.save({
        "A1": A1,
        "w1" : w1,
        "u1": u1,
        "g1": g1,
        "h1": h1,
        "v_new1": v_new1,
        "final_state1": final_state1,
        "o1": o1
    },
    'res_official.pth'
    )
    #"""

    """
    res_official = torch.load('res_official.pth')

    A1 = res_official['A1'].to(dtype)
    w1 = res_official['w1'].to(dtype)
    u1 = res_official['u1'].to(dtype)
    h1 = res_official['h1'].to(dtype)
    v_new1 = res_official['v_new1'].to(dtype)
    final_state1 = res_official['final_state1'].to(dtype)
    o1 = res_official['o1'].to(dtype)
    g1 = res_official['g1']
    # """
    # the followings are nats related !!!
    nats_block_size = NATS_Chunk
    offset_delta = delta_offset
    cu_seqlens_nats = None

    chunk_indices_delta_nats = prepare_nats_block_indices(n_nats_blocks[..., delta_offset],
                                                          nats_block_size,
                                                          chunk_size, )
    compute_incomplete_chunk_scores = True

    # """

    g2 = chunk_local_nats_cumsum(g0, chunk_size=64,
                                 nats_block_types=nats_block_types,
                                 nats_block_indices=nats_block_indices,
                                 n_nats_blocks=n_nats_blocks,
                                 chunk_indices_op_nats=chunk_indices_delta_nats,
                                 nats_block_size=nats_block_size,
                                 offset_op=offset_delta,
                                 cu_seqlens=cu_seqlens,
                                 compute_incomplete_chunk_scores=compute_incomplete_chunk_scores
                                 )
    # obtain WY representation. u is actually the new v.

    A = chunk_scaled_dot_kkt_nats_fwd(
        k=k,
        beta=beta,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        g=g2,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        offset_op=offset_delta,
        output_dtype=torch.float32,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores

    )

    A = solve_tril_nats(
        A=A,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        n_nats_blocks=n_nats_blocks,
        nats_block_indices=nats_block_indices,
        nats_block_size=nats_block_size,
        cu_seqlens=cu_seqlens,
        offset_op=offset_delta,
        output_dtype=k.dtype,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores

    )

    # torch.save({'A': A,  'g': g2, }, "A_nats.pth")
    # """

    """

    Ag = torch.load("A_nats.pth")
    A = Ag['A']
    g2 = Ag['g']
    # import pdb
    # pdb.set_trace()
    #"""
    # """
    w, u = recompute_w_u_nats_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        g=g2,
        cu_seqlens=cu_seqlens,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
    )
    # torch.save({'w': w, 'u': u,}, "tensors.pth.pth")
    """
    wu = torch.load('tensors.pth.pth')
    w = wu['w']
    u = wu['u']
    """

    # """
    nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                          nats_block_types,
                                                          nats_block_size,
                                                          64, delta_offset)
    h2, v_new, final_state2 = chunk_gated_delta_rule_nats_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g2,
        gk=None,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
        incomplete_block_start_with_ht=True,
        nats_block_delta_offsets=nats_block_delta_offsets
    )

    # torch.save({'h2': h2, 'v_new': v_new, 'final_state2': final_state2}, "hiddent_states.pth")
    # """

    """
    hs = torch.load('hiddent_states.pth')
    h2 = hs['h2']
    v_new=hs['v_new']
    final_state2 = hs['final_state2']
    #"""

    """
    i_start = 0
    torch.set_printoptions(sci_mode=False)

    for b in range(B):
        for h in range(H):
            h_nats = h // GNAtS

            n_nats_block = n_nats_blocks[b, h_nats, delta_offset]
            i_end = i_start + triton.cdiv(n_nats_block * NATS_Chunk, chunk_size) * chunk_size
            v_new_nats = v_new[i_start:i_start + n_nats_block * NATS_Chunk, h % GNAtS]
            v_new_fla = v_new1[b,:,h][mask_[b,:, h].bool()]


            i_start = i_end

            # TODO find a proper way to check if this is correct:


            import pdb
            pdb.set_trace()

    import pdb
    pdb.set_trace()
    #"""
    # """

    from nats.ops.common.chunk_o import chunk_fwd_nats_o
    nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks, nats_block_types, nats_block_size,
                                                          chunk_size, offset_delta)

    starting_h_idx = compute_starting_idx_for_chunks(
        nats_block_indices=nats_block_indices,
        T=T,
        BT=chunk_size,
        NAtS_Block_Size=nats_block_size,
        offset_op=offset_delta
    )

    o2 = chunk_fwd_nats_o(
        q=q,
        k=k,
        w=w,
        v=v_new,
        h=h2,
        g=g2,
        g_gamma=None,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        nats_block_op_offsets=nats_block_delta_offsets,
        starting_h_idx=starting_h_idx,
        scale=scale,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        offset_op=offset_delta,
        compute_incomplete_block_scores=compute_incomplete_chunk_scores,
        vg_is_stored_with_varlen=True,
        incomplete_block_start_with_ht=True,
    )
    o11 = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h1,
        g=g2,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    import pdb
    pdb.set_trace()

    for b in range(B):
        for h0 in range(H):
            h_nats = h0 // GNAtS
            for i in range(triton.cdiv(T, chunk_size)):
                nats_block_type = nats_block_types[b, i, h0, offset_delta]
                o2_vanilla = o11[b, i * 64: i * 64 + 64, h0]
                o1_nats = o2[b, i * 64: i * 64 + 64, h0]
                if nats_block_type > 0:
                    diff = o1_nats - o2_vanilla
                    print(True)
                    print(f'diff with valid v_new: {diff.abs().max()} at b={b}, h={h0}, i={i}')
                else:

                    xq = q[b, i * 64: i * 64 + 64, h0]
                    xk = k[b, i * 64: i * 64 + 64, h0]
                    xg = g2[b, i * 64: i * 64 + 64, h0]
                    b_A = xq @ xk.T
                    msk = torch.arange(0, 64).cuda()[:, None] >= torch.arange(0, 64).cuda()[None, :]
                    b_A = b_A * torch.exp(xg[:, None] - xg[None, :])
                    b_A = torch.where(msk, b_A, 0)
                    w0 = w[b, i * 64: i * 64 + 64, h0]

                    diff = o1_nats - (o2_vanilla - b_A.to(h1) @ (w0 @ h1[b, i, h0]) * scale)
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
    beta = torch.randn(B, T, H, dtype=dtype, device=device).sigmoid()
    g0 = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device=device) * 20)
    # g0 = torch.zeros(B, T, H, dtype=torch.float32, device=device)

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
    output_final_state = True
    cu_seqlens = None
    mask_ = nats_block_types[..., delta_offset]
    mask_ = mask_.repeat_interleave(NATS_Chunk, 1)
    mask_ = mask_[:, :, :, None].expand(B, NATS_Chunk * T_NAtS, HNatS, GNAtS).reshape(B, NATS_Chunk * T_NAtS,
                                                                                      HNatS * GNAtS)
    mask_ = mask_[:, :T, :]
    q1 = copy.deepcopy(q)
    do1 = copy.deepcopy(do)
    dht1 = copy.deepcopy(dht)
    k1 = copy.deepcopy(k) * mask_[..., None]
    v1 = copy.deepcopy(v) * mask_[..., None]
    beta1 = copy.deepcopy(beta) * mask_  # should not be necessary here

    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum, solve_tril
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
    import math
    scale = 1 / math.sqrt(K)
    """
    g1 = chunk_local_cumsum(copy.deepcopy(g0) * mask_, chunk_size=64, cu_seqlens=None)
    g11 = chunk_local_cumsum(copy.deepcopy(g0), chunk_size=64, cu_seqlens=None)

    A1 = chunk_scaled_dot_kkt_fwd(
        k=k1,
        g=g1,
        beta=beta1,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )

    A1 = solve_tril(
        A=A1,
        cu_seqlens=cu_seqlens,
        output_dtype=k1.dtype
    )

    w1, u1 = recompute_w_u_fwd(
        k=k1,
        v=v1,
        beta=beta1,
        A=A1,
        g=g1,
        cu_seqlens=cu_seqlens,
    )
    h1, v_new1, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w1,
        u=u1,
        g=g1,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )

    dv1 = chunk_bwd_dv_local(
        q=q1,
        k=k1,
        g=g1,
        do=do1,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    dh1, dh01, dv11 = chunk_gated_delta_rule_bwd_dhu(
        q=q1,
        k=k1,
        w=w1,
        g=g11,
        h0=copy.deepcopy(h0),
        dht=dht1,
        do=do1,
        dv=dv1,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dq1, dk1, dw1, dg1 = chunk_bwd_dqkwg(
        q=q1,
        k=k,
        v=v_new1,
        w=w1,
        g=g11,
        h=h1,
        dv=dv11,
        do=do1,
        dh=dh1,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dk21, dv111, db1, dg21 = prepare_wy_repr_bwd(
        k=k1,
        v=v1,
        beta=beta1,
        g=g1,
        A=A1,
        dw=dw1,
        du=dv11,
        cu_seqlens=cu_seqlens,
    )
    torch.save({
        "A1": A1,
        "w1": w1,
        "u1": u1,
        "g1": g1,
        "g11": g11,
        "dv1": dv1,
        "dw1": dw1,
        "v_new1": v_new1,
        "dv11": dv11,
        "dh01": dh01,
        "dq1": dq1,
        "dk1": dk1,
        "h1": h1,
        "dg1": dg1,
        "dk21": dk21,
        "dh1": dh1,
        "dv111": dv111,
        "db1": db1,
        "dg21": dg21
    }, "bwd.pth"
    )
    #"""
    """
    torch.save({
        "A1": A1,
        "w1": w1,
        "u1": u1,
        "g1": g1,
        "g11": g11,
        "dv1": dv1,
        "dw1": dw1,
        "v_new1": v_new1,
        "dv11": dv11,
        "dh01": dh01,
        "dq1": dq1,
        "dk1": dk1,
        "h1": h1,
        "dg1": dg1,
        "dk21": dk21,
        "dh1" : dh1,
        "dv111": dv111,
        "db1": db1,
        "dg21": dg21
    }, "bwd.pth"
    )
    #"""

    """
    bwd_gated_delta_net = torch.load("bwd.pth")
    A1 = bwd_gated_delta_net['A1']
    w1 = bwd_gated_delta_net['w1']
    u1 = bwd_gated_delta_net['u1']
    g1 = bwd_gated_delta_net['g1']
    g11 = bwd_gated_delta_net['g11']
    dv1 = bwd_gated_delta_net['dv1']
    dv11 = bwd_gated_delta_net['dv11']
    dh01 = bwd_gated_delta_net['dh01']
    dq1 = bwd_gated_delta_net['dq1']
    dk1 = bwd_gated_delta_net['dk1']
    dg1 = bwd_gated_delta_net['dg1']
    dk21 = bwd_gated_delta_net['dk21']
    dw1 = bwd_gated_delta_net['dw1']
    dv111 = bwd_gated_delta_net['dv111']
    db1 = bwd_gated_delta_net['db1']
    dg21 = bwd_gated_delta_net['dg21']
    v_new1 = bwd_gated_delta_net['v_new1']
    h1 = bwd_gated_delta_net['h1']
    dh1 = bwd_gated_delta_net['dh1']
    # """

    """
    dq1, dk1, dw1, dg1 = chunk_bwd_dqkwg(
        q=q1,
        k=k1,
        v=v_new1,
        w=w1,
        g=g1,
        h=h1,
        dv=dv11,
        do=do1,
        dh=dh1,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    #bwd_gated_delta_net['dw1'] = dw1
    #torch.save(bwd_gated_delta_net, "bwd.pth")
    """
    """
    dh1, dh01, dv11 = chunk_gated_delta_rule_bwd_dhu(
        q=q1,
        k=k1,
        w=w1,
        g=g1,
        h0=copy.deepcopy(h0),
        dht=dht1,
        do=do1,
        dv=dv1,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    #"""

    nats_block_size = NATS_Chunk
    offset_delta = delta_offset
    cu_seqlens_nats = None

    chunk_indices_delta_nats = prepare_nats_block_indices(n_nats_blocks[..., delta_offset],
                                                          nats_block_size,
                                                          chunk_size, )
    compute_incomplete_chunk_scores = True
    nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks, nats_block_types, nats_block_size,
                                                          chunk_size, offset_delta)
    """
    g2 = chunk_local_nats_cumsum(g0, chunk_size=64,
                                 nats_block_types=nats_block_types,
                                 nats_block_indices=nats_block_indices,
                                 n_nats_blocks=n_nats_blocks,
                                 chunk_indices_op_nats=chunk_indices_delta_nats,
                                 nats_block_size=nats_block_size,
                                 offset_op=offset_delta,
                                 cu_seqlens=cu_seqlens,
                                 compute_incomplete_chunk_scores=compute_incomplete_chunk_scores
                                 )
    # obtain WY representation. u is actually the new v.

    A = chunk_scaled_dot_kkt_nats_fwd(
        k=k,
        beta=beta,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        g=g2,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        offset_op=offset_delta,
        output_dtype=torch.float32,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores

    )

    A2 = solve_tril_nats(
        A=A,
        chunk_indices_op_nats=chunk_indices_delta_nats,
        n_nats_blocks=n_nats_blocks,
        nats_block_indices=nats_block_indices,
        nats_block_size=nats_block_size,
        cu_seqlens=cu_seqlens,
        offset_op=offset_delta,
        output_dtype=k.dtype,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores

    )
    w, u = recompute_w_u_nats_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A2,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        g=g2,
        cu_seqlens=cu_seqlens,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
    )

    h2, v_new, final_state2 = chunk_gated_delta_rule_nats_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g2,
        gk=None,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        save_new_value=True,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        nats_block_delta_offsets=nats_block_delta_offsets,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores
    )

    """
    """
    torch.save({'A2': A2,
                'w': w,
                'u': u,
                'h': h2,
                'v_new': v_new,
                'g2': g2,
                }, 'gq.pth',
               )
    import pdb
    pdb.set_trace()
    #"""

    # """
    from nats.ops.common.chunk_o import chunk_fwd_nats_o
    nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks, nats_block_types, nats_block_size,
                                                          chunk_size, offset_delta)

    starting_h_idx = compute_starting_idx_for_chunks(
        nats_block_indices=nats_block_indices,
        T=T,
        BT=chunk_size,
        NAtS_Block_Size=nats_block_size,
        offset_op=offset_delta
    )
    # """
    bwd_info_nats = torch.load('gq.pth')

    g2 = bwd_info_nats['g2']

    A2 = bwd_info_nats['A2']
    w = bwd_info_nats['w']
    u = bwd_info_nats['u']
    h2 = bwd_info_nats['h']
    v_new = bwd_info_nats['v_new']
    # import pdb
    # pdb.set_trace()
    """

    #"""
    v_new_old = v_new.clone()
    dv, qdo, v_new = chunk_bwd_dv_qdo_nats_local(q=q,
                                                 k=k,
                                                 do=do,
                                                 w=w,
                                                 v_new=v_new,
                                                 h=h2,
                                                 nats_block_types=nats_block_types,
                                                 nats_block_indices=nats_block_indices,
                                                 n_nats_blocks=n_nats_blocks,
                                                 chunk_indices_op_nats=chunk_indices_delta_nats,
                                                 g=g2,
                                                 scale=scale,
                                                 nats_block_size=nats_block_size,
                                                 offset_op=offset_delta,
                                                 compute_incomplete_block_scores=compute_incomplete_chunk_scores,
                                                 pre_compute_qdo=True,
                                                 incomplete_block_start_with_ht=True
                                                 )

    # torch.save({'dv': dv, 'qdo': qdo}, 'qdo.pth')
    # """
    # """
    i_start = 0

    for b in range(B):
        for h0 in range(H):
            h_nats = h0 % GNAtS
            print(f"b: {b}, h {h0}")
            for i in range(T_NAtS):
                h_cur = h2[i_start, h_nats]
                if nats_block_types[b, i, h0, offset_delta]:
                    print(f' delta ops')

                    v_current = v_new_old[b, i * 64:(i + 1) * 64, h0]
                else:
                    print(f'non delta ops')

                    v_current = v_new_old[b, i * 64:(i + 1) * 64, h0]
                    xw = w[b, i * 64:(i + 1) * 64, h0]
                    v_current = v_current - xw @ h_cur

                v_nats = v_new[b, i * 64:(i + 1) * 64, h0]
                diff = v_nats - v_current
                print(f'diff vnats: {diff.abs().max()}')
                if nats_block_types[b, i, h0, offset_delta]:
                    i_start += 1

    i_start = 0
    for b in range(B):
        for h in range(H):
            h_nats = h % GNAtS
            nats_chunk_idx = nats_block_indices[b, :, h, offset_delta].view(-1, triton.cdiv(64,
                                                                                            nats_block_size)) * nats_block_size
            for i in range(triton.cdiv(T, 64)):
                if i == 0:
                    if nats_chunk_idx[i, 0] != 0:
                        idx_start = 0
                    else:
                        idx_start = 64
                else:
                    idx_start = nats_chunk_idx[i - 1, 0] + 64
                if idx_start >= T:
                    continue
                # if nats_chunk_idx[i+1, 0]>=T and nats_block_types[b,-1,h, offset_delta] == 0:
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
                        xg = g2[b, q_indices, h]
                        # dv_vanilla = (torch.where(msk, xk@xq.T, 0).float() * scale * torch.exp(xgq[None, :] - xg[:, None])).to(dtype) @ xdo
                        qdo_vanilla = (xq.T.float() * scale * torch.exp(xg[None, :])).to(dtype) @ xdo

                        xw = w[b, q_indices, h]
                        xdv = dv[b, q_indices, h]
                        qdo_vanilla -= xw.T @ xdv

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
    dh, dh0, dv01 = chunk_gated_delta_rule_nats_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g2,
        h0=h0,
        dht=dht,
        do=do,
        dv=dv,
        qdo=qdo,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
        nats_block_delta_offsets=nats_block_delta_offsets,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        nats_block_size=nats_block_size,
        offset_delta=offset_delta,
        compute_incomplete_chunk_scores=compute_incomplete_chunk_scores,
    )
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
    for b in range(B):
        for ih in range(H):
            h_nats = ih % GNAtS
            nats_chunk_idx = nats_block_indices[b, :, ih, offset_delta]
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
    import pdb
    pdb.set_trace()
    """
    data_dh = torch.load('dh.pth')
    dh = data_dh['dh']
    dv01 = data_dh['dv01']
    i_start = 0
    #"""
    """
    for b in range(B):
        for h in range(H):
            h_nats = h // GNAtS

            n_nats_block = n_nats_blocks[b, h_nats, delta_offset]
            i_end = i_start + triton.cdiv(n_nats_block * NATS_Chunk, chunk_size) * chunk_size
            dv_new_nats = dv01[i_start:i_start + n_nats_block * NATS_Chunk, h % GNAtS]
            dv_new_fla = dv11[b, :, h][mask_[b, :, h].bool()]
            diff = dv_new_nats - dv_new_fla

            dv_nats = dv[i_start:i_start + n_nats_block * NATS_Chunk, h % GNAtS]
            dv1_fla = dv1[b, :, h][mask_[b, :, h].bool()]

            i_start = i_end

            # TODO find a proper way to check if this is correct:

            import pdb
            pdb.set_trace()
    """
    # """
    starting_h_idx = compute_starting_idx_for_chunks(
        nats_block_indices=nats_block_indices,
        T=T,
        BT=chunk_size,
        NAtS_Block_Size=nats_block_size,
        offset_op=offset_delta
    )
    nats_block_op_offsets = prepare_nats_chunk_offsets(n_nats_blocks, nats_block_types, nats_block_size,
                                                       chunk_size, offset_delta)
    dq, dk, dw, dg, d_nats = chunk_bwd_nats_dqkwg(
        q=q, k=k, v=v_new, do=do, h=h2, dh=dh, nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices, n_nats_blocks=n_nats_blocks,
        chunk_indices_op_nats=chunk_indices_delta_nats, g=g2, g_gamma=None,
        starting_h_idx=starting_h_idx,
        nats_block_op_offsets=nats_block_op_offsets,
        scale=scale,
        dv=dv01, w=w, nats_block_size=nats_block_size, offset_op=offset_delta,
        compute_incomplete_block_scores=compute_incomplete_chunk_scores,
        compute_dnats_for_invalid_blocks=True,

    )

    i_start = 0

    for b in range(B):
        for h0 in range(H):
            h_nats = h0 % GNAtS
            print(f"b: {b}, h {h0}")
            for i in range(T_NAtS):
                h_cur = h2[i_start, h_nats]
                dh_cur = dh[i_start, h_nats].T
                if nats_block_types[b, i, h0, offset_delta]:
                    print(f' delta ops')

                    v_current = v_new1[b, i * 64:(i + 1) * 64, h0]
                else:
                    print(f'non delta ops')

                    v_current = v_new1[b, i * 64:(i + 1) * 64, h0]
                    xw = w[b, i * 64:(i + 1) * 64, h0]
                    v_current = v_current - xw @ h_cur

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

    assert dg1.dtype == torch.float32, "dg should be fp32"
    dg1 = chunk_local_cumsum(dg1, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    test_bwd()
