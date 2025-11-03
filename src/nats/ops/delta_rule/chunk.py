# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from nats.ops.common.chunk_delta_h import chunk_gated_delta_rule_nats_bwd_dhu, chunk_gated_delta_rule_nats_fwd_h
# from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from nats.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_nats_fwd
from nats.ops.delta_rule.wy_fast import prepare_wy_repr_nats_bwd, prepare_wy_repr_nats_fwd, recompute_w_u_nats_fwd
from nats.ops.common.chunk_o import chunk_bwd_nats_dqkwg, chunk_bwd_dv_qdo_nats_local, chunk_fwd_nats_o
from fla.ops.utils import chunk_local_cumsum
from nats.ops.utils import solve_tril_nats
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks


# TODO in the current implementation, we use O_{t} = Q_{t} @ h_{t0} + p_t @ (v_t - w_t @ h_t0)
#  However, an alternative could also be   O_{t} = Q_{t} @ h_{t0} + p_t @ (v_t), which requires a
#  different fwd and bwd implementation...

def chunk_delta_rule_nats_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_nats_fwd(
        k=k,
        beta=beta,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
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
        g=None,
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
    )
    o = chunk_fwd_nats_o(
        q=q,
        k=k,
        v=v_new if incomplete_block_start_with_ht else u,
        w=w,
        h=h,
        g=None,
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
        keep_wu_as_kv=keep_wu_as_kv,
    )
    return o, A, final_state


def chunk_delta_rule_nats_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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
        chunk_size: int = 64,
):
    assert keep_wu_as_kv == compute_incomplete_chunk_scores or compute_dnats_for_invalid_blocks
    w, u = recompute_w_u_nats_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        chunk_indices_delta_nats=chunk_indices_delta_nats,
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
        g=None,
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
                                                 g=None,
                                                 scale=scale,
                                                 nats_block_size=nats_block_size,
                                                 offset_op=offset_delta,
                                                 compute_incomplete_block_scores=compute_incomplete_chunk_scores,
                                                 incomplete_block_start_with_ht=incomplete_block_start_with_ht,
                                                 )

    dh, dh0, dv = chunk_gated_delta_rule_nats_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
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
        compute_dnats_for_invalid_blocks=compute_dnats_for_invalid_blocks,
    )

    dq, dk, dw, _, d_nats = chunk_bwd_nats_dqkwg(
        q=q, k=k, v=v_new, do=do, h=h, dh=dh, nats_block_types=nats_block_types,
        nats_block_indices=nats_block_indices,
        nats_block_op_offsets=nats_block_delta_offsets,
        n_nats_blocks=n_nats_blocks,
        starting_h_idx=starting_h_idx_delta,
        chunk_indices_op_nats=chunk_indices_delta_nats, g=None, g_gamma=None,
        dv=dv, w=w, cu_seqlens=cu_seqlens, cu_seqlens_nats=cu_seqlens_nats,
        nats_block_size=nats_block_size, scale=scale, offset_op=offset_delta,
        compute_incomplete_block_scores=compute_incomplete_chunk_scores,
        compute_dnats_for_invalid_blocks=compute_dnats_for_invalid_blocks,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
    )
    dk2, dv, db = prepare_wy_repr_nats_bwd(
        k=k,
        v=v,
        beta=beta,
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
        keep_wu_as_kv=keep_wu_as_kv,
    )
    dk.add_(dk2)
    return dq, dk, dv, db, dh0, d_nats


class ChunkDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
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

        o, A, final_state = chunk_delta_rule_nats_fwd(
            q=q,
            k=k,
            v=v,
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
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, cu_seqlens_nats,
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
            q, q_rstd, k, k_rstd, v, beta,
            A, initial_state, cu_seqlens, cu_seqlens_nats, nats_block_types,
            nats_block_indices, n_nats_blocks, chunk_indices_delta_nats,
            nats_block_delta_offsets, starting_h_idx
        ) = ctx.saved_tensors
        dq, dk, dv, db, dh0, d_nats = chunk_delta_rule_nats_bwd(
            q=q,
            k=k,
            v=v,
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
        return dq.to(q), dk.to(k), dv.to(v), db.to(beta), None, dh0, None, None, None


@torch.compiler.disable
def chunk_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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
    o, final_state = ChunkDeltaRuleFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel
    )
    return o, final_state
