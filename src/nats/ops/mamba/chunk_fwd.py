# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import warnings
from typing import Optional

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks
from nats.ops.mamba.chunk_h import chunk_mamaba_nats_fwd_h, chunk_mamba_nats_bwd_dhu
from nats.ops.mamba.chunk_o import chunk_mamba_fwd_nats_o, chunk_bwd_dv_qdo_nats_local, chunk_bwd_nats_dqkwg
from nats.ops.mamba.chunk_cumsum import chunk_nats_cumsum_fwd, chunk_nats_cumsum_bwd
from nats.ops.utils.cumsum import chunk_cumsum_non_gated_chunks

from fla.ops.utils import chunk_local_cumsum
from nats.ops.utils import solve_tril_nats
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks


def chunk_mamba_inference_fwd(
        X: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor,
        g: torch.Tensor,
        nats_block_types: torch.Tensor,
        op_type_last_chunk: torch.Tensor | None,
        nats_block_indices: torch.Tensor,
        n_nats_blocks: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        # we note that in this case, the initial_state should always be the state of the current chunks
        output_final_state: bool,
        chunk_indices_mamba_nats: torch.Tensor,
        nats_block_mamba_offsets: torch.Tensor,
        starting_h_idx_mamba: torch.Tensor,
        decay_for_non_mamba_blocks: bool = True,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_nats: Optional[torch.LongTensor] = None,
        nats_block_size: int = 64,
        offset_mamba: int = 0,
        compute_incomplete_chunk_scores: bool = False,
        incomplete_block_start_with_ht: bool = True,
        chunk_size: int = 64,
        n_tokens_in_init_chunk: int = 0,
        compute_o: bool=True,
):
    assert nats_block_size >= chunk_size, "The current implementaion only allows one nats block within each delta " \
                                          "computaionl chunk!!!"
    if compute_o:
        t_q = C.shape[1]
        t_k = B.shape[1]
        assert t_q + n_tokens_in_init_chunk == t_k
    else:
        t_k = B.shape[1]
        assert t_k == chunk_size

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

    # We note that here only the parts that are mamba parts are updated with v
    # For the other part, it we have incomplete_block_start_with_ht, they will be updated within the
    # chunk_fwd_nats_o function,
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
    if compute_o:
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
        # TODO we need to consider decay here!
        if output_final_state:
            batch_size, num_heads = n_nats_blocks.shape[:2]
            h_chunk_start = h[torch.cumsum(n_nats_blocks[..., offset_mamba].flatten(), 0) - 1]
            h_chunk_start = h_chunk_start.view(batch_size, num_heads, B.shape[-1], X.shape[-1])
            n_tokens_in_last_chunk = t_k % nats_block_size

            if decay_for_non_mamba_blocks:
                if n_tokens_in_last_chunk == 0 or nats_block_types.shape[1] <= 1:
                    h_chunk_start = h_chunk_start * torch.exp(
                        g[:, -n_tokens_in_last_chunk - 1].view(batch_size, num_heads, 1, 1))
                else:
                    # in this case, we need to check if the last complete chunk is a gdn block.If it is,
                    # then the hidden state is already decayed and we do not need decay this term.
                    decay = torch.where(
                        nats_block_types[:,-2,:, offset_mamba] > 0,
                        1.,
                        torch.exp(g[:, -n_tokens_in_last_chunk - 1])
                    )
                    h_chunk_start = h_chunk_start * decay.view(batch_size, num_heads, 1, 1)

            if n_tokens_in_last_chunk == 0:
                assert op_type_last_chunk is not None
                h_chunk_start = torch.where((op_type_last_chunk[..., offset_mamba]>0).view(batch_size, num_heads, 1, 1),
                                            final_state,
                                            h_chunk_start)
        else:
            final_state = None
            h_chunk_start = None

    else:
        # if we do not compute o, we need to ensure that
        o = None
        h_chunk_start = final_state

    return o, final_state, h_chunk_start
