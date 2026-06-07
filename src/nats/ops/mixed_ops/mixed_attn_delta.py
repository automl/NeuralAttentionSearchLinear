from dataclasses import dataclass
import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous, input_guard

from nats.ops.attns.attns import parallel_attn_nats_fwd, parallel_attn_nats_bwd, parallel_attn_bwd_chunk_size
from nats.ops.delta_rule.chunk import chunk_delta_rule_nats_fwd, chunk_delta_rule_nats_bwd

from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks

CHUNK_SIZE = 64

_attn_streams: dict[int, torch.cuda.Stream] = {}


def _get_attn_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _attn_streams:
        _attn_streams[idx] = torch.cuda.Stream(device=idx)
    return _attn_streams[idx]


@dataclass
class RunIncompleteBlock:
    attn: bool = False
    delta_net: bool = False


all_incomplete_ops: dict[str, RunIncompleteBlock] = {
    'all': RunIncompleteBlock(attn=True,
                              delta_net=True),
    'attn': RunIncompleteBlock(attn=True,
                               delta_net=False),
    'delta_net': RunIncompleteBlock(attn=False,
                                    delta_net=True)
}


class NAtSMixedAttentionDelta(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx,
                q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
                q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
                initial_state_delta: torch.Tensor,
                beta: torch.Tensor,
                nats_block_types: torch.Tensor,
                n_nats_blocks: torch.Tensor,  # n_nats_blocks is acquired by attn_types.int().sum(1)
                scale_attn, scale_lattn, cu_seqlens, cu_seqlens_nats,
                nats_block_size: int,
                ops_for_incomplete_chunks: str = 'delta_net',
                output_final_state_gated_delta: bool = False,
                compute_dnats_for_invalid_blocks_attn: bool = False,
                compute_dnats_for_invalid_blocks_linear_att: bool = True,
                incomplete_block_start_with_ht: bool = True,
                lattn_use_qk_l2norm_in_kernel: bool = True,
                ):
        incomplete_block_strategy = all_incomplete_ops[ops_for_incomplete_chunks]

        if not incomplete_block_strategy.delta_net:
            if torch.is_grad_enabled():
                keep_wu_as_kv = compute_dnats_for_invalid_blocks_linear_att
            else:
                keep_wu_as_kv = False
        else:
            keep_wu_as_kv = True

        incomplete_block_start_with_ht = incomplete_block_strategy.delta_net and incomplete_block_start_with_ht

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

        # TODO make this a dict? check if it can pass torch compile's check
        OFFSET_ATTN = 0
        OFFSET_DELTA = 1


        o_attn, lse_attn, _ = parallel_attn_nats_fwd(
            q_attn, k_attn, v_attn, nats_block_types, nats_block_indices,
            None, scale_attn, NAtS_block_size=nats_block_size,
            offset_attn=0, compute_incomplete_chunk_scores=incomplete_block_strategy.attn,
            is_causal=True, store_msk=False,
        )

        # delta-net on current stream (overlaps with attn kernel above)
        if lattn_use_qk_l2norm_in_kernel:
            q_lattn, q_rstd_lattn = l2norm_fwd(q_lattn)
            k_lattn, k_rstd_lattn = l2norm_fwd(k_lattn)
        else:
            q_rstd_lattn, k_rstd_lattn = None, None

        chunk_indices_delta_nats = prepare_nats_block_indices(n_nats_blocks[..., OFFSET_DELTA],
                                                              nats_block_size,
                                                              CHUNK_SIZE, )
        nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                              nats_block_types,
                                                              nats_block_size,
                                                              CHUNK_SIZE, OFFSET_DELTA)
        starting_h_idx_gated_delta = compute_starting_idx_for_chunks(
            nats_block_indices=nats_block_indices,
            T=T,
            BT=CHUNK_SIZE,
            NAtS_Block_Size=nats_block_size,
            offset_op=OFFSET_DELTA
        )

        o_gated_delta, A_delta, final_state_gated_delta = chunk_delta_rule_nats_fwd(
            q=q_lattn,
            k=k_lattn,
            v=v_lattn,
            beta=beta,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=scale_lattn,
            initial_state=initial_state_delta,
            output_final_state=output_final_state_gated_delta,
            chunk_indices_delta_nats=chunk_indices_delta_nats,
            nats_block_delta_offsets=nats_block_delta_offsets,
            starting_h_idx_delta=starting_h_idx_gated_delta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=nats_block_size,
            offset_delta=OFFSET_DELTA,
            compute_incomplete_chunk_scores=incomplete_block_strategy.delta_net,
            incomplete_block_start_with_ht=incomplete_block_start_with_ht,
            keep_wu_as_kv=keep_wu_as_kv,
        )

        ctx.save_for_backward(
            q_attn, k_attn, v_attn, o_attn, lse_attn,
            q_lattn, q_rstd_lattn, k_lattn, k_rstd_lattn, v_lattn, beta, A_delta,
            initial_state_delta, chunk_indices_delta_nats, nats_block_delta_offsets, starting_h_idx_gated_delta,
            cu_seqlens, cu_seqlens_nats,
            nats_block_types, nats_block_indices, n_nats_blocks,
        )

        ctx.scale_attn = scale_attn
        ctx.scale_lattn = scale_lattn

        ctx.nats_block_size = nats_block_size
        ctx.lattn_use_qk_l2norm_in_kernel = lattn_use_qk_l2norm_in_kernel

        ctx.OFFSET_ATTN = OFFSET_ATTN
        ctx.OFFSET_DELTA = OFFSET_DELTA

        ctx.ops_for_incomplete_chunks = ops_for_incomplete_chunks
        ctx.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        ctx.compute_dnats_for_invalid_blocks_linear_att = compute_dnats_for_invalid_blocks_linear_att
        ctx.incomplete_block_start_with_ht = incomplete_block_start_with_ht
        ctx.keep_wu_as_kv = keep_wu_as_kv
        ctx.incomplete_block_strategy = incomplete_block_strategy

        return o_gated_delta, o_attn, final_state_gated_delta

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
            ctx,
            do_gated_delta: torch.Tensor,
            do_attn: torch.Tensor,
            dht_gated_net: torch.Tensor
    ):

        (
            q_attn, k_attn, v_attn, o_attn, lse_attn,
            q_lattn, q_rstd_lattn, k_lattn, k_rstd_lattn, v_lattn, beta, A_delta,
            initial_state_delta, chunk_indices_delta_nats, nats_block_delta_offsets, starting_h_idx_gated_delta,
            cu_seqlens, cu_seqlens_nats,
            nats_block_types, nats_block_indices, n_nats_blocks,
        ) = ctx.saved_tensors

        # Pre-compute attn chunk indices on current_stream (before fork) to eliminate
        # the GPU-CPU sync that prepare_nats_block_indices would cause on attn_stream.
        _use_dkdv_path = not (ctx.compute_dnats_for_invalid_blocks_attn)
        if _use_dkdv_path:
            _bt_dkdv = parallel_attn_bwd_chunk_size(
                q_attn.shape[-1], ctx.nats_block_size, q_attn.device.index
            )
            chunk_indices_attn_nats = prepare_nats_block_indices(
                n_nats_blocks[..., ctx.OFFSET_ATTN], ctx.nats_block_size, _bt_dkdv
            )
        else:
            chunk_indices_attn_nats = None

        dq_attn, dk_attn, dv_attn, dg_attn, dnats_attn = parallel_attn_nats_bwd(
            q_attn, k_attn, v_attn, o_attn, nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            g_cumsum=None,
            lse=lse_attn,
            do=do_attn,
            scale=ctx.scale_attn,
            NAtS_block_size=ctx.nats_block_size,
            OFFSET_ATTN=ctx.OFFSET_ATTN,
            compute_dnats_for_invalid_blocks_attn=ctx.compute_dnats_for_invalid_blocks_attn,
            compute_incomplete_chunk_scores=ctx.incomplete_block_strategy.attn,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            is_causal=True,
            chunk_indices_attn_nats=chunk_indices_attn_nats,
        )

        # Delta bwd on current stream (overlaps with attn dq + dkdv kernels above)
        dq_delta, dk_delta, dv_delta, db, dh0_delta, dnats_delta = chunk_delta_rule_nats_bwd(
            q=q_lattn,
            k=k_lattn,
            v=v_lattn,
            beta=beta,
            A=A_delta,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=ctx.scale_lattn,
            initial_state=initial_state_delta,
            do=do_gated_delta,
            dht=dht_gated_net,
            chunk_indices_delta_nats=chunk_indices_delta_nats,
            nats_block_delta_offsets=nats_block_delta_offsets,
            starting_h_idx_delta=starting_h_idx_gated_delta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=ctx.nats_block_size,
            offset_delta=ctx.OFFSET_DELTA,
            compute_incomplete_chunk_scores=ctx.incomplete_block_strategy.delta_net,
            compute_dnats_for_invalid_blocks=ctx.compute_dnats_for_invalid_blocks_linear_att,
            incomplete_block_start_with_ht=ctx.incomplete_block_start_with_ht,
            keep_wu_as_kv=ctx.keep_wu_as_kv,
        )

        if ctx.lattn_use_qk_l2norm_in_kernel:
            dq_delta = l2norm_bwd(q_lattn, q_rstd_lattn, dq_delta)
            dk_delta = l2norm_bwd(k_lattn, k_rstd_lattn, dk_delta)

        # TODO if we want a uniform dv, we might need some adjustments here!!!
        #  check how to solve problems for dg for transformer
        # this should be adjusted based on our choice, we need to check how to solve this
        dnats = torch.cat([dnats_attn.unsqueeze(-1), dnats_delta.unsqueeze(-1)], dim=-1)
        # TODO if we have other linear attn types, we need to adjust them here!
        return dq_attn, dk_attn, dv_attn, \
               dq_delta, dk_delta, dv_delta, \
               dh0_delta, \
               db, dnats, None, \
               None, None, None, None, \
               None, None, None, None, None, None, None, None


def nats_mixed_attn_delta(
        q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
        q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
        initial_state_delta: torch.Tensor,
        beta: torch.Tensor,
        nats_block_types: torch.Tensor,
        n_nats_blocks: torch.Tensor,  # n_nats_blocks is acquired by attn_types.int().sum(1)
        scale_attn=None, scale_lattn=None,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size: int = 64,
        ops_for_incomplete_chunks: str = 'gated_delta_net',
        output_final_state_gated_delta: bool = False,
        compute_dnats_for_invalid_blocks_attn: bool = False,  # TODO this is not activated yet, we need to fix that!
        compute_dnats_for_invalid_blocks_linear_att: bool = True,
        incomplete_block_start_with_ht: bool = True,
        use_g_for_attn: bool = False,
        lattn_use_qk_l2norm_in_kernel: bool = True,
):
    if scale_attn is None:
        scale_attn = k_attn.shape[-1] ** -0.5
    if scale_lattn is None:
        scale_lattn = k_lattn.shape[-1] ** -0.5

    o_gated_delta_net, o_attn, gated_delta_ht = NAtSMixedAttentionDelta.apply(
        q_attn, k_attn, v_attn, q_lattn, k_lattn, v_lattn,
        initial_state_delta, beta,
        nats_block_types, n_nats_blocks, scale_attn,
        scale_lattn, cu_seqlens, cu_seqlens_nats,
        nats_block_size, ops_for_incomplete_chunks,
        output_final_state_gated_delta,
        compute_dnats_for_invalid_blocks_attn,
        compute_dnats_for_invalid_blocks_linear_att,
        incomplete_block_start_with_ht,
        lattn_use_qk_l2norm_in_kernel
    )
    return o_gated_delta_net, o_attn, gated_delta_ht


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

    beta = torch.randn(BATCH, T, HDELTA, dtype=dtype, device=device, requires_grad=True).sigmoid()
    g0 = F.logsigmoid(torch.rand(BATCH, T, HDELTA, dtype=torch.float32, device=device, requires_grad=True) * 20)

    q = torch.nn.Parameter(q, requires_grad=True)
    k = torch.nn.Parameter(k, requires_grad=True)
    v = torch.nn.Parameter(v, requires_grad=True)
    beta = torch.nn.Parameter(beta, requires_grad=True)
    g0 = torch.nn.Parameter(g0, requires_grad=True)
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

    out, _ = nats_mixed_attn_delta(
        q_attn=q_attn, k_attn=k_attn, v_attn=v_attn,
        q_lattn=q_lattn, k_lattn=k_lattn, v_lattn=v_lattn,
        initial_state_delta=None,
        g=g0, beta=beta, nats_block_types=nats_block_types, n_nats_blocks=n_nats_blocks,
        scale_attn=k_attn.shape[-1] ** -0.5, scale_lattn=k_lattn.shape[-1] ** -0.5,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size=NATS_block_size,
        ops_for_incomplete_chunks='all',
        compute_dnats_for_invalid_blocks_attn=False,
        compute_dnats_for_invalid_blocks_linear_att=False
    )

    loss = (out ** 2)
    loss.sum().backward()
    import pdb
    pdb.set_trace()



if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    test_mixed_attn()