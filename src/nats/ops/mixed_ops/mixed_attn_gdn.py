from dataclasses import dataclass
import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous, input_guard
from fla.ops.utils.cumsum import chunk_global_cumsum

from nats.ops.attns.attns import parallel_attn_nats_fwd, parallel_attn_nats_bwd
from nats.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_nats_fwd, chunk_gated_delta_rule_nats_bwd

from nats.ops.nats_util import prepare_nats_block_indices, prepare_nats_chunk_offsets, compute_starting_idx_for_chunks

CHUNK_SIZE = 64
all_nats_ops = ['gated_delta', 'gated_delta_net']


@dataclass
class RunIncompleteBlock:
    attn: bool = False
    gated_delta_net: bool = False


all_incomplete_ops: dict[str, RunIncompleteBlock] = {
    'all': RunIncompleteBlock(attn=True,
                              gated_delta_net=True),
    'attn': RunIncompleteBlock(attn=True,
                               gated_delta_net=False),
    'gated_delta_net': RunIncompleteBlock(attn=False,
                                          gated_delta_net=True)
}




@torch.compile
class NAtSMixedAttentionGDN(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx,
                q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
                q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
                initial_state_gated_delta: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                nats_block_types: torch.Tensor,
                n_nats_blocks: torch.Tensor,  # n_nats_blocks is acquired by attn_types.int().sum(1)
                scale_attn, scale_lattn, cu_seqlens, cu_seqlens_nats,
                nats_block_size: int,
                attn_sw_size: int | None = None,
                ops_for_incomplete_chunks: str = 'gated_delta_net',
                output_final_state_gated_delta: bool = False,
                compute_dnats_for_invalid_blocks_attn: bool = False,
                compute_dnats_for_invalid_blocks_linear_att: bool = True,
                decay_for_non_gdn_blocks: bool=False,
                incomplete_block_start_with_ht: bool = True,
                use_g_for_attn: bool = True,
                lattn_use_qk_l2norm_in_kernel: bool = True,
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
        if not incomplete_block_strategy.gated_delta_net:
            keep_wu_as_kv = compute_dnats_for_invalid_blocks_linear_att
        else:
            keep_wu_as_kv = True

        incomplete_block_start_with_ht = incomplete_block_strategy.gated_delta_net and incomplete_block_start_with_ht

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
        OFFSET_GATED_DELTA_NET = 1

        # attns
        o_attn, lse_attn, _ = parallel_attn_nats_fwd(
            q_attn, k_attn, v_attn, nats_block_types, nats_block_indices,
            g_cumsum_attn, scale_attn, NAtS_block_size=nats_block_size,
            sliding_window_size=attn_sw_size,
            offset_attn=0, compute_incomplete_chunk_scores=incomplete_block_strategy.attn if attn_sw_size is not None else False,
            is_causal=True, store_msk=False,
        )

        # gated delta net
        if lattn_use_qk_l2norm_in_kernel:
            q_lattn, q_rstd_lattn = l2norm_fwd(q_lattn)
            k_lattn, k_rstd_lattn = l2norm_fwd(k_lattn)
        else:
            q_rstd_lattn, k_rstd_lattn = None, None


        chunk_indices_delta_nats = prepare_nats_block_indices(n_nats_blocks[..., OFFSET_GATED_DELTA_NET],
                                                              nats_block_size,
                                                              CHUNK_SIZE, )
        nats_block_delta_offsets = prepare_nats_chunk_offsets(n_nats_blocks,
                                                              nats_block_types,
                                                              nats_block_size,
                                                              CHUNK_SIZE, OFFSET_GATED_DELTA_NET)
        starting_h_idx_gated_delta = compute_starting_idx_for_chunks(
            nats_block_indices=nats_block_indices,
            T=T,
            BT=CHUNK_SIZE,
            NAtS_Block_Size=nats_block_size,
            offset_op=OFFSET_GATED_DELTA_NET
        )

        g_gated_delta, o_gated_delta, A_gated_delta, final_state_gated_delta = chunk_gated_delta_rule_nats_fwd(
            q=q_lattn,
            k=k_lattn,
            v=v_lattn,
            g=g,
            beta=beta,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=scale_lattn,
            initial_state=initial_state_gated_delta,
            output_final_state=output_final_state_gated_delta,
            chunk_indices_delta_nats=chunk_indices_delta_nats,
            nats_block_delta_offsets=nats_block_delta_offsets,
            starting_h_idx_delta=starting_h_idx_gated_delta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=nats_block_size,
            offset_delta=OFFSET_GATED_DELTA_NET,
            compute_incomplete_chunk_scores=incomplete_block_strategy.gated_delta_net,
            incomplete_block_start_with_ht=incomplete_block_start_with_ht,
            decay_for_non_gdn_blocks=decay_for_non_gdn_blocks,
            keep_wu_as_kv=keep_wu_as_kv,
        )

        ctx.save_for_backward(
            q_attn, k_attn, v_attn, o_attn, g_cumsum_attn, lse_attn,
            q_lattn, q_rstd_lattn, k_lattn, k_rstd_lattn, v_lattn, g_gated_delta, beta, A_gated_delta,
            initial_state_gated_delta, chunk_indices_delta_nats, nats_block_delta_offsets, starting_h_idx_gated_delta,
            cu_seqlens, cu_seqlens_nats,
            nats_block_types, nats_block_indices, n_nats_blocks,
        )

        ctx.scale_attn = scale_attn
        ctx.scale_lattn = scale_lattn

        ctx.nats_block_size = nats_block_size
        ctx.attn_sw_size = attn_sw_size
        ctx.lattn_use_qk_l2norm_in_kernel = lattn_use_qk_l2norm_in_kernel

        ctx.OFFSET_ATTN = OFFSET_ATTN
        ctx.OFFSET_GATED_DELTA = OFFSET_GATED_DELTA_NET

        ctx.ops_for_incomplete_chunks = ops_for_incomplete_chunks
        ctx.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        ctx.compute_dnats_for_invalid_blocks_linear_att = compute_dnats_for_invalid_blocks_linear_att
        ctx.incomplete_block_start_with_ht = incomplete_block_start_with_ht
        ctx.keep_wu_as_kv = keep_wu_as_kv
        ctx.incomplete_block_strategy = incomplete_block_strategy
        ctx.decay_for_non_gdn_blocks = decay_for_non_gdn_blocks

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
            q_attn, k_attn, v_attn, o_attn, g_cumsum_attn, lse_attn,
            q_lattn, q_rstd_lattn, k_lattn, k_rstd_lattn, v_lattn, g_gated_delta, beta, A_gated_delta,
            initial_state_gated_delta, chunk_indices_delta_nats, nats_block_delta_offsets, starting_h_idx_gated_delta,
            cu_seqlens, cu_seqlens_nats,
            nats_block_types, nats_block_indices, n_nats_blocks,
        ) = ctx.saved_tensors

        # attns
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
        )

        dq_gated_delta, dk_gated_delta, dv_gated_delta, db, dg_gated_delta, dh0_gated_delta, dnats_gated_delta \
            = chunk_gated_delta_rule_nats_bwd(
            q=q_lattn,
            k=k_lattn,
            v=v_lattn,
            g=g_gated_delta,
            beta=beta,
            A=A_gated_delta,
            nats_block_types=nats_block_types,
            nats_block_indices=nats_block_indices,
            n_nats_blocks=n_nats_blocks,
            scale=ctx.scale_lattn,
            initial_state=initial_state_gated_delta,
            do=do_gated_delta,
            dht=dht_gated_net,
            chunk_indices_delta_nats=chunk_indices_delta_nats,
            nats_block_delta_offsets=nats_block_delta_offsets,
            starting_h_idx_delta=starting_h_idx_gated_delta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_nats=cu_seqlens_nats,
            nats_block_size=ctx.nats_block_size,
            offset_delta=ctx.OFFSET_GATED_DELTA,
            compute_incomplete_chunk_scores=ctx.incomplete_block_strategy.gated_delta_net,
            compute_dnats_for_invalid_blocks=ctx.compute_dnats_for_invalid_blocks_linear_att,
            incomplete_block_start_with_ht=ctx.incomplete_block_start_with_ht,
            keep_wu_as_kv=ctx.keep_wu_as_kv,
            decay_for_non_gdn_blocks=ctx.decay_for_non_gdn_blocks,
        )
        # TODO if we want a uniform dv, we might need some adjustments here!!!
        #  check how to solve problems for dg for transformer
        # this should be adjusted based on our choice, we need to check how to solve this
        dnats = torch.cat([dnats_attn.unsqueeze(-1), dnats_gated_delta.unsqueeze(-1)], dim=-1)

        if ctx.lattn_use_qk_l2norm_in_kernel:
            dq_gated_delta = l2norm_bwd(q_lattn, q_rstd_lattn, dq_gated_delta)
            dk_gated_delta = l2norm_bwd(k_lattn, k_rstd_lattn, dk_gated_delta)
        # TODO if we have other linear attn types, we need to adjust them here!
        return dq_attn, dk_attn, dv_attn, \
               dq_gated_delta, dk_gated_delta, dv_gated_delta, \
               dh0_gated_delta, \
               dg_gated_delta, db, dnats, None, \
               None, None, None, None, \
               None, None, None, None, None, None, None, None, None, None


def nats_mixed_attn_gdn(
        q_attn: torch.Tensor, k_attn: torch.Tensor, v_attn: torch.Tensor,
        q_lattn: torch.Tensor, k_lattn: torch.Tensor, v_lattn: torch.Tensor,
        initial_state_gated_delta: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        nats_block_types: torch.Tensor,
        n_nats_blocks: torch.Tensor,  # n_nats_blocks is acquired by attn_types.int().sum(1)
        scale_attn=None, scale_lattn=None,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size: int = 64,
        attn_sw_size: int | None = None,
        ops_for_incomplete_chunks: str = 'gated_delta_net',
        output_final_state_gated_delta: bool = False,
        compute_dnats_for_invalid_blocks_attn: bool = False,  # TODO this is not activated yet, we need to fix that!
        compute_dnats_for_invalid_blocks_linear_att: bool = True,
        decay_for_non_gdn_blocks:bool=False,
        incomplete_block_start_with_ht: bool = True,
        use_g_for_attn: bool = False,
        lattn_use_qk_l2norm_in_kernel: bool = True,
):
    if scale_attn is None:
        scale_attn = k_attn.shape[-1] ** -0.5
    if scale_lattn is None:
        scale_lattn = k_lattn.shape[-1] ** -0.5

    o_gated_delta_net, o_attn, gated_delta_ht = NAtSMixedAttentionGDN.apply(
        q_attn, k_attn, v_attn, q_lattn, k_lattn, v_lattn,
        initial_state_gated_delta, g, beta,
        nats_block_types, n_nats_blocks, scale_attn,
        scale_lattn, cu_seqlens, cu_seqlens_nats,
        nats_block_size, attn_sw_size, ops_for_incomplete_chunks,
        output_final_state_gated_delta,
        compute_dnats_for_invalid_blocks_attn,
        compute_dnats_for_invalid_blocks_linear_att,
        decay_for_non_gdn_blocks,
        incomplete_block_start_with_ht,
        use_g_for_attn, lattn_use_qk_l2norm_in_kernel
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

    o_gated_delta_net,  o_attn, gated_delta_ht = nats_mixed_attn_gdn(
        q_attn=q_attn, k_attn=k_attn, v_attn=v_attn,
        q_lattn=q_lattn, k_lattn=k_lattn, v_lattn=v_lattn,
        initial_state_gated_delta=None,
        g=g0, beta=beta, nats_block_types=nats_block_types, n_nats_blocks=n_nats_blocks,
        scale_attn=k_attn.shape[-1] ** -0.5, scale_lattn=k_lattn.shape[-1] ** -0.5,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size=NATS_block_size,
        ops_for_incomplete_chunks='attn',
        compute_dnats_for_invalid_blocks_attn=False,
        compute_dnats_for_invalid_blocks_linear_att=True
    )

    loss = (o_gated_delta_net ** 2) + o_attn.view(o_gated_delta_net.shape) ** 2
    loss.sum().backward()
    import pdb
    pdb.set_trace()



if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # test_compute_h()
    test_mixed_attn()

