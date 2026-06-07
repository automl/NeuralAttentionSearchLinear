from dataclasses import dataclass
import torch
from flash_attn import flash_attn_varlen_func, flash_attn_func
from torch.nn import functional as F
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous, input_guard
from fla.ops.utils.cumsum import chunk_global_cumsum

from nats.ops.attns.attns import parallel_attn_nats_fwd, parallel_attn_nats_bwd, parallel_attn_bwd_chunk_size
from nats.ops.attns.attn_utils import repeat_masks
#from nats.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_nats_fwd, chunk_gated_delta_rule_nats_bwd
from nats.ops.mamba.chunk import chunk_mamba2_fwd, chunk_mamba2_bwd
from nats.ops.mamba.chunk_fwd import chunk_mamba_inference_fwd
from nats.ops.mamba.fused_recurrent import fused_recurrent_mamba_nats_fwd

from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks

CHUNK_SIZE = 64
all_nats_ops = ['gated_delta', 'gated_delta_net', 'mamba']

_attn_streams: dict[int, torch.cuda.Stream] = {}


def _get_attn_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _attn_streams:
        _attn_streams[idx] = torch.cuda.Stream(device=idx)
    return _attn_streams[idx]


@dataclass
class RunIncompleteBlock:
    attn: bool = False
    mamba: bool = False


all_incomplete_ops: dict[str, RunIncompleteBlock] = {
    'all': RunIncompleteBlock(attn=True,
                              mamba=True),
    'attn': RunIncompleteBlock(attn=True,
                               mamba=False),
    'mamba': RunIncompleteBlock(attn=False,
                                mamba=True)
}




class NAtSMixedAttentionMamba(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx,
                q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
                q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
                initial_state_mamba: torch.Tensor,
                g: torch.Tensor,
                nats_block_types: torch.Tensor,
                n_nats_blocks: torch.Tensor,  # n_nats_blocks is acquired by attn_types.int().sum(1)
                scale_attn, scale_lattn, cu_seqlens, cu_seqlens_nats,
                nats_block_size: int,
                attn_sw_size: int | None = None,
                ops_for_incomplete_chunks: str = 'all',
                output_final_state_mamba: bool = False,
                compute_dnats_for_invalid_blocks_attn: bool = False,
                compute_dnats_for_invalid_blocks_linear_att: bool = True,
                decay_for_non_mamba_blocks: bool=False,
                incomplete_block_start_with_ht: bool = True,
                use_g_for_attn: bool = True,
                ):
        incomplete_block_strategy = all_incomplete_ops[ops_for_incomplete_chunks]
        """
        # TODO check how to check if this function is forwarded under no_grad env!
        if not incomplete_block_strategy.gated_delta_net:
            if torch.is_grad_enabled():
                keep_wu_as_kv = compute_dnats_for_invalid_blocks_linear_att
            else:
                keep_wu_as_kv = False
        else:
            keep_wu_as_kv = True
        """

        incomplete_block_start_with_ht = incomplete_block_strategy.mamba and incomplete_block_start_with_ht

        nats_block_types = torch.round(nats_block_types) # TODO  we need to investigate this further...

        # TODO this is only for head first is True,
        assert nats_block_size >= CHUNK_SIZE
        TNAtS = nats_block_types.shape[1]
        T = q_lattn.shape[1]
        nats_block_indices = torch.where(
            nats_block_types == 1.,
            torch.arange(TNAtS, device=nats_block_types.device, dtype=torch.int32).view(1, -1, 1, 1), TNAtS
        )
        nats_block_indices = nats_block_indices.sort(1)[0]

        if n_nats_blocks is None:
            n_nats_blocks = nats_block_types.int().sum(1)
        if use_g_for_attn:
            RCP_LN2: float = 1.4426950216
            g_cumsum_attn = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2) if g is not None else None
        else:
            g_cumsum_attn = None

        # TODO make this a dict? check if it can pass torch compile's check
        OFFSET_ATTN = 0
        OFFSET_MAMBA = 1

        # Launch attn on secondary stream; linear-attn runs on current stream in parallel.
        # _attn_live pins all pre-fork tensors passed as raw pointers to the attn kernel,
        # preventing the allocator from recycling them before attn_stream finishes.
        _attn_live = (q_attn, k_attn, v_attn, nats_block_types, nats_block_indices, g_cumsum_attn)
        attn_stream = _get_attn_stream(q_attn.device)
        attn_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(attn_stream):
            o_attn, lse_attn, _ = parallel_attn_nats_fwd(
                q_attn, k_attn, v_attn, nats_block_types, nats_block_indices,
                g_cumsum_attn, scale_attn, NAtS_block_size=nats_block_size,
                sliding_window_size=attn_sw_size,
                offset_attn=0, compute_incomplete_chunk_scores=incomplete_block_strategy.attn if attn_sw_size is None else False,
                is_causal=True, store_msk=False,
            )

        chunk_indices_mamba_nats = prepare_nats_block_indices(n_nats_blocks[..., OFFSET_MAMBA],
                                                              nats_block_size,
                                                              CHUNK_SIZE, )
        nats_block_mamba_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                              nats_block_types,
                                                              nats_block_size,
                                                              CHUNK_SIZE, OFFSET_MAMBA)
        starting_h_idx_mamba = compute_starting_idx_for_chunks(
            nats_block_indices=nats_block_indices,
            T=T,
            BT=CHUNK_SIZE,
            NAtS_Block_Size=nats_block_size,
            offset_op=OFFSET_MAMBA
        )
        o_mamba, g_mamba, final_state_mamba = chunk_mamba2_fwd(
            X=v_lattn,
            B=k_lattn,
            C=q_lattn,
            g=g,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=scale_lattn,
            initial_state=initial_state_mamba,
            output_final_state=output_final_state_mamba,
            chunk_indices_mamba_nats=chunk_indices_mamba_nats,
            nats_block_mamba_offsets=nats_block_mamba_offsets,
            starting_h_idx_mamba=starting_h_idx_mamba,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=nats_block_size,
            offset_mamba=OFFSET_MAMBA,
            compute_incomplete_chunk_scores=incomplete_block_strategy.mamba,
            incomplete_block_start_with_ht=incomplete_block_start_with_ht,
            decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
        )

        # Sync: ensure attn kernel has completed before saving its outputs
        torch.cuda.current_stream().wait_stream(attn_stream)
        del _attn_live

        ctx.save_for_backward(
            q_attn, k_attn, v_attn, o_attn, g_cumsum_attn, lse_attn,
            q_lattn, k_lattn, v_lattn, g_mamba, 
            initial_state_mamba, chunk_indices_mamba_nats, nats_block_mamba_offsets, starting_h_idx_mamba,
            cu_seqlens, cu_seqlens_nats,
            nats_block_types, nats_block_indices, n_nats_blocks,
        )

        ctx.scale_attn = scale_attn
        ctx.scale_lattn = scale_lattn

        ctx.nats_block_size = nats_block_size
        ctx.attn_sw_size = attn_sw_size

        ctx.OFFSET_ATTN = OFFSET_ATTN
        ctx.OFFSET_MAMBA = OFFSET_MAMBA
        ctx.output_final_state = output_final_state_mamba

        ctx.ops_for_incomplete_chunks = ops_for_incomplete_chunks
        ctx.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        ctx.compute_dnats_for_invalid_blocks_linear_att = compute_dnats_for_invalid_blocks_linear_att
        ctx.incomplete_block_start_with_ht = incomplete_block_start_with_ht
        ctx.incomplete_block_strategy = incomplete_block_strategy
        ctx.decay_for_non_mamba_blocks = decay_for_non_mamba_blocks

        return o_mamba, o_attn, final_state_mamba

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
            ctx,
            do_mamba: torch.Tensor,
            do_attn: torch.Tensor,
            dht_mamba: torch.Tensor
    ):

        (
           q_attn, k_attn, v_attn, o_attn, g_cumsum_attn, lse_attn,
           q_lattn, k_lattn, v_lattn, dA_cumsum, 
           initial_state_mamba, chunk_indices_mamba_nats, nats_block_mamba_offsets, starting_h_idx_mamba,
           cu_seqlens, cu_seqlens_nats,
           nats_block_types, nats_block_indices, n_nats_blocks,
        ) = ctx.saved_tensors

        # Pre-compute attn chunk indices on current_stream (before fork) to eliminate
        # the GPU-CPU sync that prepare_nats_block_indices would cause on attn_stream.
        _use_dkdv_path = not (ctx.attn_sw_size is not None or ctx.compute_dnats_for_invalid_blocks_attn)
        if _use_dkdv_path:
            _bt_dkdv = parallel_attn_bwd_chunk_size(
                q_attn.shape[-1], ctx.nats_block_size, q_attn.device.index
            )
            chunk_indices_attn_nats = prepare_nats_block_indices(
                n_nats_blocks[..., ctx.OFFSET_ATTN], ctx.nats_block_size, _bt_dkdv
            )
        else:
            chunk_indices_attn_nats = None

        # Launch attn bwd on secondary stream; gated-delta bwd runs on current stream in parallel.
        _attn_live = (q_attn, k_attn, v_attn, o_attn, g_cumsum_attn, lse_attn,
                      nats_block_types, nats_block_indices, n_nats_blocks, chunk_indices_attn_nats)
        attn_stream = _get_attn_stream(q_attn.device)
        attn_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(attn_stream):
            dq_attn, dk_attn, dv_attn, dg_attn, dnats_attn = parallel_attn_nats_bwd(
                q_attn, k_attn, v_attn, o_attn, nats_block_types=nats_block_types,
                nats_block_indices=nats_block_indices,
                n_nats_blocks=n_nats_blocks,
                g_cumsum=g_cumsum_attn,
                lse=lse_attn,
                do=do_attn,
                scale=ctx.scale_attn,
                NAtS_block_size=ctx.nats_block_size,
                sliding_window_size=ctx.attn_sw_size,
                OFFSET_ATTN=ctx.OFFSET_ATTN,
                compute_dnats_for_invalid_blocks_attn=ctx.compute_dnats_for_invalid_blocks_attn,
                compute_incomplete_chunk_scores=ctx.incomplete_block_strategy.attn if ctx.attn_sw_size is not None else False,
                cu_seqlens=cu_seqlens,
                cu_seqlens_nats=cu_seqlens_nats,
                is_causal=True,
                chunk_indices_attn_nats=chunk_indices_attn_nats,
            )

        # Gated-delta bwd on current stream (overlaps with attn dq + dkdv kernels above)
        dq_mamba, dk_mamba, dv_mamba, dg, dh0_mamba, dnats_mamba \
            = chunk_mamba2_bwd(
            X=v_lattn,
            B=k_lattn,
            C=q_lattn,
            dA_cumsum=dA_cumsum,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=ctx.scale_lattn,
            initial_state=initial_state_mamba,
            do=do_mamba,
            dht=dht_mamba,
            chunk_indices_mamba_nats=chunk_indices_mamba_nats,
            nats_block_mamba_offsets=nats_block_mamba_offsets,
            starting_h_idx_mamba=starting_h_idx_mamba,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=ctx.nats_block_size,
            offset_mamba=ctx.OFFSET_MAMBA,
            compute_incomplete_chunk_scores=ctx.incomplete_block_strategy.mamba,
            compute_dnats_for_invalid_blocks=ctx.compute_dnats_for_invalid_blocks_linear_att,
            incomplete_block_start_with_ht=ctx.incomplete_block_start_with_ht,
            decay_for_non_mamba_blocks=ctx.decay_for_non_mamba_blocks,
        )

        # Sync: wait for attn bwd to complete before using its outputs
        torch.cuda.current_stream().wait_stream(attn_stream)
        del _attn_live

        # TODO if we want a uniform dv, we might need some adjustments here!!!
        #  check how to solve problems for dg for transformer
        # this should be adjusted based on our choice, we need to check how to solve this
        dnats = torch.cat([dnats_attn.unsqueeze(-1), dnats_mamba.unsqueeze(-1)], dim=-1)
        # TODO if we have other linear attn types, we need to adjust them here!

        return dq_attn, dk_attn, dv_attn, \
               dq_mamba, dk_mamba, dv_mamba, \
               dh0_mamba, \
               dg, dnats, \
               None, None, None, None, \
               None, None, None, None, None, None, None, None, None, None


def nats_mixed_attn_mamba(
        q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
        q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
        initial_state_mamba: torch.Tensor,
        g: torch.Tensor,
        nats_block_types: torch.Tensor,
        n_nats_blocks: torch.Tensor,  # n_nats_blocks is acquired by attn_types.int().sum(1)
        scale_attn=None, scale_lattn=None,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size: int = 64,
        attn_sw_size: int | None = None,
        ops_for_incomplete_chunks: str = 'all',
        output_final_state_mamba: bool = False,
        compute_dnats_for_invalid_blocks_attn: bool = False,  # TODO this is not activated yet, we need to fix that!
        compute_dnats_for_invalid_blocks_linear_att: bool = True,
        decay_for_non_mamba_blocks:bool=False,
        incomplete_block_start_with_ht: bool = True,
        use_g_for_attn: bool = False,
):
    if scale_attn is None:
        scale_attn = k_attn.shape[-1] ** -0.5
    if scale_lattn is None:
        scale_lattn = 1

    o_mamba, o_attn, mamba_ht = NAtSMixedAttentionMamba.apply(
        q_attn, k_attn, v_attn, q_lattn, k_lattn, v_lattn,
        initial_state_mamba, g,
        nats_block_types, n_nats_blocks, scale_attn,
        scale_lattn, cu_seqlens, cu_seqlens_nats,
        nats_block_size, attn_sw_size, ops_for_incomplete_chunks,
        output_final_state_mamba,
        compute_dnats_for_invalid_blocks_attn,
        compute_dnats_for_invalid_blocks_linear_att,
        decay_for_non_mamba_blocks,
        incomplete_block_start_with_ht,
        use_g_for_attn
    )
    return o_mamba, o_attn, mamba_ht


@torch.compile
def one_step_attn(q_attn:torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor, attn_msk: torch.Tensor, scale_attn:float):
    mask_flatten = attn_msk.flatten()
    q_states_flatten = rearrange(q_attn, 'b t (h g) d -> (b h t) g  d', h=attn_msk.shape[1])
    k_states_flatten = rearrange(k_attn, 'b t (h g) d -> (b h t) g  d', h=attn_msk.shape[1])[mask_flatten]
    v_states_flatten = rearrange(v_attn, 'b t (h g) d -> (b h t) g  d', h=attn_msk.shape[1])[mask_flatten]
    n_pad = v_states_flatten.shape[-1] - q_states_flatten.shape[-1]
    q_states_flatten = F.pad(q_states_flatten, (0, n_pad))
    k_states_flatten = F.pad(k_states_flatten, (0, n_pad))
    cu_seqlens_q = torch.arange(1 + len(q_states_flatten), dtype=torch.int32,
            device=q_states_flatten.device)
    n_valid_tokens = attn_msk.sum(-1).flatten()

    cu_seqlens_k = F.pad(torch.cumsum(n_valid_tokens.flatten(), -1, dtype=torch.int32), (1, 0))
    attn_output = flash_attn_varlen_func(
            q_states_flatten,
            k_states_flatten,
            v_states_flatten,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=torch.max(n_valid_tokens),
            softmax_scale=scale_attn,
            )
    o_attn = rearrange(attn_output, '(b h t) g d -> b t (h g) d', h=attn_msk.shape[1], t=1)
    return o_attn


def nats_mixed_attn_mamba_chunk_inference(
        q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
        q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
        initial_state_mamba_start: torch.Tensor, # TODO this should also be hte part
        g: torch.Tensor,
        #g_cumsum: torch.Tensor, # TODO do we need this?
        nats_block_types: torch.Tensor,
        op_type_last_chunk: torch.Tensor | None,
        n_nats_blocks: torch.Tensor,  # n_nats_blocks is acquired by attn_types.int().sum(1)
        n_tokens_in_current_block:int = 0,
        scale_attn=None, scale_lattn=None,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size: int = 64,
        ops_for_incomplete_chunks: str = 'all',
        output_final_state_mamba: bool = True,
        decay_for_non_mamba_blocks:bool=True,
        incomplete_block_start_with_ht: bool = True,
        use_g_for_attn: bool = False,
):
    incomplete_block_strategy = all_incomplete_ops[ops_for_incomplete_chunks]
    if scale_attn is None:
        scale_attn = k_attn.shape[-1] ** -0.5
    if scale_lattn is None:
        scale_lattn = k_lattn.shape[-1] ** -0.5

    TNAtS = nats_block_types.shape[1]
    T = q_lattn.shape[1]
    nats_block_indices = torch.where(
        nats_block_types == 1.,
        torch.arange(TNAtS, device=nats_block_types.device, dtype=torch.int32).view(1, -1, 1, 1), TNAtS
    )
    nats_block_indices = nats_block_indices.sort(1)[0]

    if n_nats_blocks is None:
        n_nats_blocks = nats_block_types.int().sum(1)
    if use_g_for_attn:
        RCP_LN2: float = 1.4426950216
        g_cumsum_attn = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2) if g is not None else None
    else:
        g_cumsum_attn = None

    # TODO make this a dict? check if it can pass torch compile's check
    OFFSET_ATTN = 0
    OFFSET_MAMBA = 1

    # attns

    o_attn, lse_attn, msk = parallel_attn_nats_fwd(
        q_attn, k_attn, v_attn, nats_block_types, nats_block_indices,
        g_cumsum_attn, scale_attn, NAtS_block_size=nats_block_size,
        offset_attn=OFFSET_ATTN,
        compute_incomplete_chunk_scores=incomplete_block_strategy.attn,
        is_causal=True, store_msk=False,
    )

    chunk_indices_mamba_nats = prepare_nats_block_indices(n_nats_blocks[..., OFFSET_MAMBA],
                                                          nats_block_size,
                                                          CHUNK_SIZE, )
    nats_block_mamba_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                          nats_block_types,
                                                          nats_block_size,
                                                          CHUNK_SIZE, OFFSET_MAMBA)

    starting_h_idx_mamba = compute_starting_idx_for_chunks(
        nats_block_indices=nats_block_indices,
        T=T,
        BT=CHUNK_SIZE,
        NAtS_Block_Size=nats_block_size,
        offset_op=OFFSET_MAMBA
    )

    o_mamba, mamba_final_state, mamba_chunk_start = chunk_mamba_inference_fwd(
        X=v_lattn,
        B=k_lattn,
        C=q_lattn,
        g=g,
        nats_block_types=nats_block_types,
        op_type_last_chunk=op_type_last_chunk,
        nats_block_indices=nats_block_indices,
        n_nats_blocks=n_nats_blocks,
        scale=scale_lattn,
        initial_state=initial_state_mamba_start,
        output_final_state=output_final_state_mamba,
        chunk_indices_mamba_nats=chunk_indices_mamba_nats,
        nats_block_mamba_offsets=nats_block_mamba_offsets,
        starting_h_idx_mamba=starting_h_idx_mamba,
        cu_seqlens=cu_seqlens,
        cu_seqlens_nats=cu_seqlens_nats,
        nats_block_size=nats_block_size,
        offset_mamba=OFFSET_MAMBA,
        compute_incomplete_chunk_scores=incomplete_block_strategy.mamba,
        incomplete_block_start_with_ht=incomplete_block_start_with_ht,
        decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
    )
    return o_mamba, o_attn, mamba_final_state, mamba_chunk_start


def nats_mixed_attn_mamba_recurrent(
        q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
        q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
        attn_msk: torch.Tensor,
        initial_state_mamba: torch.Tensor,
        initial_state_mamba_chunk_start: torch.Tensor,
        g: torch.Tensor,
        g_cumsum: torch.Tensor,
        nats_block_types: torch.Tensor,
        n_tokens_in_current_block:int = 0,
        scale_attn=None, scale_lattn=None,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size: int = 64,
        ops_for_incomplete_chunks: str = 'all',
        output_final_state_mamba: bool = False,
        decay_for_non_mamba_blocks:bool=False,
        incomplete_block_start_with_ht: bool = True,
        use_g_for_attn: bool = False, # TODO this is also required!!!
):
    # TODO consider the case where decay still happens with inactivate gdn blocks!
    if scale_attn is None:
        scale_attn = k_attn.shape[-1] ** -0.5
    if scale_lattn is None:
        scale_lattn = k_lattn.shape[-1] ** -0.5

    incomplete_block_strategy = all_incomplete_ops[ops_for_incomplete_chunks]
    OFFSET_ATTN = 0
    OFFSET_MAMBA = 1

    stream_softmax_attn = torch.cuda.Stream()
    stream_lattn = torch.cuda.Stream()
    with torch.cuda.stream(stream_softmax_attn):
        attn_msk = repeat_masks(attn_msk, q_attn.shape[2] // attn_msk.shape[1])
        o_attn = torch.nn.functional.scaled_dot_product_attention(
            q_attn.transpose(1,2), k_attn.transpose(1,2), v_attn.transpose(1,2), attn_mask=attn_msk, scale=scale_attn
        ).transpose(1,2).contiguous()

    with torch.cuda.stream(stream_lattn):
        if incomplete_block_strategy.mamba:
            # in this case, we simply do a one step recurrent fwd
            o_mamba, final_state, initial_state_in_current_block = fused_recurrent_mamba_nats_fwd(
                q_lattn, k_lattn, v_lattn, g=g, g_cumsum=g_cumsum,
                nats_block_types=nats_block_types,
                n_tokens_in_current_block=n_tokens_in_current_block,
                scale=scale_lattn,
                initial_state=initial_state_mamba,
                initial_state_in_current_block=initial_state_mamba_chunk_start,
                cu_seqlens=cu_seqlens,
                cu_seqlens_nats=cu_seqlens_nats,
                nats_block_size=nats_block_size,
                offset_op=OFFSET_MAMBA,
                decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
                only_update_hidden_states=False,
                output_final_state=True,
                update_hs_for_each_iter=True,
            )
        else:
            n_tokens_first = nats_block_size - n_tokens_in_current_block
            # in this case, we first directly compute Q out
            o_mamba = torch.empty(*q_lattn.shape[:2], *v_lattn.shape[2:], device=v_lattn.device, dtype=q_lattn.dtype)
            o_mamba[:, :n_tokens_first], final_state, initial_state_in_current_block = fused_recurrent_mamba_nats_fwd(
                q_lattn[:, :n_tokens_first], k_lattn, v_lattn, g=g[:, :n_tokens_first].contiguous() + g_cumsum,
                nats_block_types=nats_block_types,
                n_tokens_in_current_block=n_tokens_in_current_block,
                scale=scale_lattn,
                initial_state=initial_state_mamba,
                initial_state_in_current_block=initial_state_mamba_chunk_start,
                cu_seqlens=cu_seqlens,
                cu_seqlens_nats=cu_seqlens_nats,
                nats_block_size=nats_block_size,
                offset_op=OFFSET_MAMBA,
                decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
                only_update_hidden_states=False,
                output_final_state=True,
            )
            # we now use parallel form to update the hidden states
            n_iters = (n_tokens_in_current_block + q_lattn.shape[1] - 1) // nats_block_size
            for i in range(n_iters):
                i_start = i * nats_block_size
                i_end = (i + 1) * nats_block_size

                _, final_state, initial_state_in_current_block = fused_recurrent_mamba_nats_fwd(
                    q=None, k=k_lattn[:, i_start:i_end].contiguous(), v=v_lattn[:, i_start:i_end].contiguous(),
                    g=g[:, :n_tokens_first].sum(1,  keepdim=True) + g_cumsum,
                    nats_block_types=nats_block_types[:, [i]].contiguous(),
                    output_final_state=output_final_state_mamba,
                    scale=scale_lattn,
                    initial_state=initial_state_mamba,
                    initial_state_in_current_block=initial_state_mamba_chunk_start,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_nats=cu_seqlens_nats,
                    nats_block_size=nats_block_size,
                    offset_op=OFFSET_MAMBA,
                    decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
                    only_update_hidden_states=True,
                )
                i_q_start = (i + 1) * nats_block_size - n_tokens_in_current_block
                i_q_end = (i + 2) * nats_block_size - n_tokens_in_current_block
                o_mamba[:, i_q_start:i_q_end], final_state, initial_state_in_current_block  = fused_recurrent_mamba_nats_fwd(
                    q_lattn[:, i_q_start:i_q_end].contiguous(), k_lattn, v_lattn,
                    g=g[:, i_q_start:i_q_end].contiguous(),
                    nats_block_types=nats_block_types,
                    n_tokens_in_current_block=n_tokens_in_current_block,
                    scale=scale_lattn,
                    initial_state=initial_state_mamba,
                    initial_state_in_current_block=initial_state_mamba_chunk_start,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_nats=cu_seqlens_nats,
                    nats_block_size=nats_block_size,
                    offset_op=OFFSET_MAMBA,
                    decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
                    only_update_hidden_states=False,
                    output_final_state=True,
                )
            if (n_tokens_in_current_block + q_lattn.shape[1]) % nats_block_size == 0:
                i_start = n_tokens_in_current_block + q_lattn.shape[1] - nats_block_size
                # we need to update the final state for the last time:
                _, final_state, initial_state_in_current_block = fused_recurrent_mamba_nats_fwd(
                    q=None, k=k_lattn[:, i_start:].contiguous(), v=v_lattn[:, i_start:].contiguous(),
                    g=g[:, i_start:].contiguous(), 
                    nats_block_types=nats_block_types[:, [-1]].contiguous(),
                    output_final_state=output_final_state_mamba,
                    scale=scale_lattn,
                    initial_state=initial_state_mamba,
                    initial_state_in_current_block=initial_state_mamba_chunk_start,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_nats=cu_seqlens_nats,
                    nats_block_size=nats_block_size,
                    offset_op=OFFSET_MAMBA,
                    decay_for_non_mamba_blocks=decay_for_non_mamba_blocks,
                    only_update_hidden_states=True,
                )

    return o_mamba, o_attn, final_state, initial_state_in_current_block


def test_mixed_attn():
    from nats.ops.gated_delta_rule.chunk import chunk_bwd_nats_dqkwg, chunk_gated_delta_rule_nats_bwd_dhu
    BATCH = 2

    HDELTA = 2

    HATTN = 4
    HNATS = 2

    T = 512
    N_OPTs = 2
    NATS_block_size = 64
    DATTN = 64
    DGATED = 128
    V_EXPAND = 2
    TNAtS = T // NATS_block_size
    torch.manual_seed(0)
    device = torch.device('cuda')
    from nats.utils import check_fp16_dtype
    from torch.nn import functional as F

    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16
    dtype = torch.float16

    q = torch.randn((BATCH, T, HDELTA * DGATED), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, T, HDELTA * DGATED), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, T, HDELTA * DGATED * V_EXPAND), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn(BATCH, TNAtS, HNATS * N_OPTs, device=device, dtype=dtype)
    A = torch.empty(HNATS, dtype=torch.float32, device=torch.device('cuda')).uniform_(0, 16)
    A = torch.log(A)
    A = -torch.exp(A.float()) / 100
    dt = torch.randn(BATCH, T, HNATS, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HNATS, device=device, dtype=dtype)

    q = torch.nn.Parameter(q, requires_grad=True)
    k = torch.nn.Parameter(k, requires_grad=True)
    v = torch.nn.Parameter(v, requires_grad=True)
    A = torch.nn.Parameter(A, requires_grad=True)
    dt = torch.nn.Parameter(dt, requires_grad=True)
    dt_bias = torch.nn.Parameter(dt_bias, requires_grad=True)
    logits = torch.nn.Parameter(logits, requires_grad=True)

    q_attn = q.view(BATCH, T, HATTN, DATTN)
    k_attn = k.view(BATCH, T, HATTN, DATTN)
    v_attn = v.view(BATCH, T, HATTN, DATTN * V_EXPAND)

    q_lattn = q.view(BATCH, T, HDELTA, DGATED)
    k_lattn = k.view(BATCH, T, HDELTA, DGATED)
    v_lattn = v.view(BATCH, T, HDELTA, DGATED * V_EXPAND)

    logits_ = logits.view(BATCH, TNAtS, HNATS, N_OPTs)
    nats_block_types = torch.nn.functional.gumbel_softmax(logits_, dim=-1, hard=True)
    nats_block_types[:,-1] = 1
    n_nats_blocks = nats_block_types.int().sum(1)

    o_mamba,  o_attn, mamba_h = nats_mixed_attn_mamba(
        q_attn=q_attn, k_attn=k_attn, v_attn=v_attn,
        q_lattn=q_lattn, k_lattn=k_lattn, v_lattn=v_lattn,
        initial_state_mamba=None,
        A=A, dt=dt, dt_bias=dt_bias,
        nats_block_types=nats_block_types, n_nats_blocks=n_nats_blocks,
        scale_attn=k_attn.shape[-1] ** -0.5, scale_lattn=1,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size=NATS_block_size,
        ops_for_incomplete_chunks='all',
        compute_dnats_for_invalid_blocks_attn=False,
        compute_dnats_for_invalid_blocks_linear_att=True
    )

    loss = (o_mamba ** 2) + o_attn.view(o_mamba.shape) ** 2
    loss.sum().backward()
    import pdb
    pdb.set_trace()



if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    test_mixed_attn()

