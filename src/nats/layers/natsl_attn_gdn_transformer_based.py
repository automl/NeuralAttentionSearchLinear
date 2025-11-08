# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
import math
from functools import partial
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from transformers.utils import logging

from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask
from fla.ops.utils.pooling import mean_pooling

from nats.ops.mixed_ops.mixed_attn_gdn import nats_mixed_attn_gdn


@torch.compile
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


@torch.compile
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

from torch.nn.functional import gumbel_softmax

def hard_softmax(
    logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    r"""
    directly generating the options for each block, this implementation follows torch's gumbel_softmax function
    but directly use the input logits as soft values,
    """
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


if TYPE_CHECKING:
    from fla.models.utils import Cache


logger = logging.get_logger(__name__)


class NeuralAttentionSearchLinearAttentionBased(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_lattn_heads: int = 16,
        num_kv_heads: Optional[int] = None,
        num_nats_head: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        n_ops: int = 2,
        nats_block_size: int = 64,
        nats_block_agg_type: str = 'mean',
        nats_sample_strategy: str = 'argmax',
        ops_for_incomplete_chunks: str = 'gated_delta_net',
        mode: str = 'chunk',
        allow_neg_eigval: bool = False,
        lattns_use_silu: bool=False,
        outputs_are_wighted: bool = False,
        compute_dnats_for_invalid_blocks_attn: bool = False,
        compute_dnats_for_invalid_blocks_linear_att: bool = False,
        incomplete_block_start_with_ht: bool = True,
        attn_apply_pos_encoding: bool = True,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_lattn_heads = num_lattn_heads
        assert num_lattn_heads <= num_heads and num_heads % num_lattn_heads == 0
        if num_nats_head is None:
            num_nats_head = num_lattn_heads

        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.head_dim_lattn = self.hidden_size // self.num_lattn_heads
        self.n_ops = n_ops

        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        #for gated delta net
        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        self.compute_dnats_for_invalid_blocks_linear_att = compute_dnats_for_invalid_blocks_linear_att
        self.incomplete_block_start_with_ht = incomplete_block_start_with_ht

        self.ops_for_incomplete_chunks = ops_for_incomplete_chunks
        self.lattns_use_silu = lattns_use_silu

        self.nats_block_size = nats_block_size
        self.num_nats_head = num_nats_head

        #if flash_attn_func is None:
        #    raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.a_proj = nn.Linear(hidden_size, self.num_lattn_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_lattn_heads, bias=False)

        if nats_block_agg_type == 'mean':
            self.pooling_func = mean_pooling
        else:
            raise NotImplementedError(f"Pooling system {nats_block_agg_type} is not supported!")
        self.nats_block_agg_type = nats_block_agg_type
        self.nats_sample_strategy = nats_sample_strategy
        if self.nats_sample_strategy == 'gumble':
            self.nats_sample_func = partial(gumbel_softmax, hard=True, dim=-1)
        else:
            self.nats_sample_func = partial(hard_softmax, dim=-1)
        self.nats_layer = nn.Linear(hidden_size, (self.n_ops * self.num_nats_head), bias=False)

        self.outputs_are_wighted = outputs_are_wighted
        if self.outputs_are_wighted:
            self.nats_out_weights_layer = nn.Linear(hidden_size, self.n_ops * self.num_nats_head, bias=False)

        A = torch.empty(self.num_lattn_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_lattn_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.attn_apply_pos_encoding = attn_apply_pos_encoding
        if attn_apply_pos_encoding:
            self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        else:
            self.rotary = None
        self.usg_for_attn = False  # TODO this can also be true, but we need to adjust the g for attns!

        self.attn_fraction = 0


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        batch_size, q_len, dim = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode

        hs_reduced = self.pooling_func(
            hidden_states.view(batch_size, q_len, 1, dim),
            chunk_size=self.nats_block_size,
            cu_seqlens=None
        )
        hs_reduced = hs_reduced.view(batch_size, hs_reduced.shape[1], dim)
        nats_op_types = self.nats_layer(hs_reduced)
        nats_op_logits = rearrange(nats_op_types, '... (h d) -> ... h d', d=self.n_ops)
        nats_op_types = self.nats_sample_func(nats_op_logits)
        # The last block is set 1 for all the operations. This does not influence the output values
        nats_op_types[:, -1] = 1
        n_nats_blocks = nats_op_types.int().sum(1)
        self.attn_fraction = (n_nats_blocks.float()[..., 0] / nats_op_types.shape[1]).mean(0).detach()

        batch_size, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.lattns_use_silu:
            q_lattn = F.silu(q)
            k_lattn = F.silu(k)
            v_lattn = F.silu(v)
        else:
            q_lattn = q
            k_lattn = k
            v_lattn = v

        q_lattn, k_lattn, v_lattn = map(
            lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim_lattn), (q_lattn, k_lattn, v_lattn))


        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        if self.attn_apply_pos_encoding:
            cu_seqlens = kwargs.get('cu_seqlens', None)

            seqlen_offset, max_seqlen = 0, q_len
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                max_seqlen = q.shape[1] + seqlen_offset

                if attention_mask is not None:
                    # to deliminate the offsets of padding tokens
                    seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                    max_seqlen = q.shape[1] + max(seqlen_offset)

            if self.max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, self.max_position_embeddings)
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        if mode == 'chunk':
            o_gated_delta, o_attn, recurrent_state_gated_delta = nats_mixed_attn(
                q_attn=q, k_attn=k, v_attn=v,
                q_lattn=q_lattn, k_lattn=k_lattn, v_lattn=v_lattn,
                initial_state_gated_delta=None, # TODO this needs ot be adjusted
                g=g, beta=beta,
                nats_block_types=nats_op_types,
                n_nats_blocks=n_nats_blocks,
                scale_attn=k.shape[-1] ** -0.5, scale_lattn=k_lattn.shape[-1] ** -0.5,
                cu_seqlens=None, cu_seqlens_nats=None,
                nats_block_size=self.nats_block_size,
                ops_for_incomplete_chunks=self.ops_for_incomplete_chunks,
                output_final_state_gated_delta=use_cache,
                compute_dnats_for_invalid_blocks_attn=self.compute_dnats_for_invalid_blocks_attn,
                compute_dnats_for_invalid_blocks_linear_att=self.compute_dnats_for_invalid_blocks_linear_att,
                incomplete_block_start_with_ht=self.incomplete_block_start_with_ht,
                use_g_for_attn=self.usg_for_attn,
                lattn_use_qk_l2norm_in_kernel=True,
            )
            if self.outputs_are_wighted:
                o_weights = self.nats_out_weights_layer(hidden_states).unflatten(-1, (self.num_nats_head, self.n_ops))
                o = o_gated_delta * o_weights[..., [1]] + o_attn.view(o_gated_delta.shape) * o_weights[..., [0]]
            else:
                o = o_gated_delta + o_attn.view(o_gated_delta.shape)
        elif mode == 'fused_recurrent':
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values



def test_mixed_attn():
    torch.manual_seed(0)
    device = torch.device('cuda')
    from nats.utils import check_fp16_dtype
    from torch.nn import functional as F

    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16

    layer = NeuralAttentionSearchLinearAttentionBased(hidden_size=512, num_heads=8, num_lattn_heads=4,
                                        ops_for_incomplete_chunks='all',
                                        compute_dnats_for_invalid_blocks_linear_att=False,
                                        compute_dnats_for_invalid_blocks_attn=False
                                        ).cuda()
    input_data = torch.randn((2, 512, 512)).cuda()
    out = layer(input_data)[0]
    label = torch.rand_like(out)

    loss = ((out - label)**2).sum()
    loss.backward()
    import pdb
    pdb.set_trace()


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
    n_nats_blocks = nats_block_types.int().sum(1)

    out, _ = nats_mixed_attn(
        q_attn=q_attn, k_attn=k_attn, v_attn=v_attn,
        q_lattn=q_lattn, k_lattn=k_lattn, v_lattn=v_lattn,
        initial_state_gated_delta=None,
        g=g0, beta=beta, nats_block_types=nats_block_types, n_nats_blocks=n_nats_blocks,
        scale_attn=k_attn.shape[-1] ** -0.5, scale_lattn=k_lattn.shape[-1] ** -0.5,
        cu_seqlens=None, cu_seqlens_nats=None,
        nats_block_size=NATS_block_size,
        ops_for_incomplete_chunks='attn',
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

