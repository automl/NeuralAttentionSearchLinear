# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask
from fla.ops.utils.pooling import mean_pooling

# from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
#from nats.ops.attns.attns import parallel_mixed_attn
#from nats.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_nats_fwd, chunk_gated_delta_rule_nats_bwd
from nats.ops.mixed_ops.mixed_attn import nats_mixed_attn

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


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


class NeuralAttentionSearchLinear(nn.Module):
    """
    The Architecture of this layer is implemented based on the GatedDeltaNet architecture

    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
            self,
            hidden_size: int = 2048,
            expand_v: float = 2,
            head_dim: int = 256,
            num_heads: int = 6,
            num_attn_heads: int = 24,
            num_v_heads: int = None,
            n_ops: int = 2,
            num_nats_head: Optional[int] = None,
            nats_block_size: int = 64,
            nats_block_agg_type: str = 'mean',
            nats_sample_strategy: str = 'argmax',
            ops_for_incomplete_chunks: str = 'gated_delta_net',
            mode: str = 'chunk',
            use_gate: bool = True,
            use_short_conv: bool = True,
            allow_neg_eigval: bool = False,
            conv_size: int = 4,
            conv_bias: bool = False,
            conv_activation: str | None = 'silu',  # TODO check if None or silu works better?
            attn_qk_norm: bool = False,
            attn_with_short_conv: bool = False,
            compute_dnats_for_invalid_blocks_attn: bool=False,
            compute_dnats_for_invalid_blocks_linear_att: bool = True,
            incomplete_block_start_with_ht: bool = True,
            attn_apply_pos_encoding: bool = True,
            attn_rope_theta: Optional[float] = 10000.,
            attn_max_position_embeddings: Optional[int] = None,
            layer_idx: int = None,
            norm_eps: float = 1e-5,
            **kwargs
    ) -> NeuralAttentionSearchLinear:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.n_ops = n_ops

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # for attns
        self.attn_qk_norm = attn_qk_norm
        self.num_attn_heads = num_attn_heads
        self.head_attn_k_dim = self.key_dim // self.num_attn_heads
        self.head_attn_v_dim = self.value_dim // self.num_attn_heads
        # TODO we need to check if this works for v heads
        self.attn_groups = num_attn_heads // num_heads
        self.num_attn_v_heads = self.attn_groups * self.num_v_heads
        self.attn_with_short_conv = attn_with_short_conv

        self.attn_rope_theta = attn_rope_theta
        self.attn_max_position_embeddings = attn_max_position_embeddings

        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.head_attn_k_dim)
            self.k_norm = RMSNorm(self.head_attn_k_dim)

        self.attn_apply_pos_encoding = attn_apply_pos_encoding

        self.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        self.compute_dnats_for_invalid_blocks_linear_att = compute_dnats_for_invalid_blocks_linear_att
        self.incomplete_block_start_with_ht = incomplete_block_start_with_ht

        if self.attn_apply_pos_encoding:
            self.rotary = RotaryEmbedding(dim=self.head_attn_k_dim, base=self.attn_rope_theta)
        else:
            self.rotary = None

        self.usg_for_attn = False  # TODO this can also be true, but we need to adjust the g for attns!

        # for nats
        if num_nats_head is None:
            num_nats_head = num_heads
        self.nats_block_size = nats_block_size
        self.num_nats_head = num_nats_head
        self.nats_block_agg_type = nats_block_agg_type
        self.ops_for_incomplete_chunks = ops_for_incomplete_chunks


        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear."
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )
        if (self.key_dim % self.num_heads) > 0:
            raise ValueError(
                f"key_dim={self.key_dim} must be divided by num_attn_heads={self.num_attn_heads}"
            )

        if self.num_attn_heads > self.num_heads and self.num_attn_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}."
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

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

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=conv_activation
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=conv_activation
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation=conv_activation
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.attn_fraction = 0

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Cache] = None, # TODO write a new cache management!!!
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        batch_size, q_len, dim = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        # TODO we need our own fun here...
        hs_reduced = self.pooling_func(
            hidden_states.view(batch_size, q_len, 1, dim),
            chunk_size=self.nats_block_size,
            cu_seqlens=cu_seqlens
        )
        hs_reduced = hs_reduced.view(batch_size, hs_reduced.shape[1], dim)
        nats_op_types = self.nats_layer(hs_reduced)
        nats_op_logits = rearrange(nats_op_types,  '... (h d) -> ... h d', d=self.n_ops)
        nats_op_types = self.nats_sample_func(nats_op_logits)
        # The last block is set 1 for all the operations. This does not influence the output values
        nats_op_types[:,-1] = 1
        n_nats_blocks = nats_op_types.int().sum(1)
        self.attn_fraction = (n_nats_blocks.float()[..., 0] / nats_op_types.shape[1]).mean(0).cpu().detach()

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if self.attn_with_short_conv:
                if last_state is not None:
                    conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
                q, conv_state_q = self.q_conv1d(
                    x=self.q_proj(hidden_states),
                    cache=conv_state_q,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens
                )
                k, conv_state_k = self.k_conv1d(
                    x=self.k_proj(hidden_states),
                    cache=conv_state_k,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens
                )
                v, conv_state_v = self.v_conv1d(
                    x=self.v_proj(hidden_states),
                    cache=conv_state_v,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens
                )
                q_attn = q
                k_attn = k
                v_attn = v
            else:
                q_attn = self.q_proj(hidden_states)
                k_attn = self.k_proj(hidden_states)
                v_attn = self.v_proj(hidden_states)
                
                q, conv_state_q = self.q_conv1d(
                    x=q_attn,
                    cache=conv_state_q,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens
                )
                k, conv_state_k = self.k_conv1d(
                    x=k_attn,
                    cache=conv_state_k,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens
                )
                v, conv_state_v = self.v_conv1d(
                    x=v_attn,
                    cache=conv_state_v,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens
                )
                
        else:
            if self.attn_with_short_conv:
                q = F.silu(self.q_proj(hidden_states))
                k = F.silu(self.k_proj(hidden_states))
                v = F.silu(self.v_proj(hidden_states))
                
                q_attn = q
                k_attn = k
                v_attn = v
            else:
                q_attn = self.q_proj(hidden_states)
                k_attn = self.k_proj(hidden_states)
                v_attn = self.v_proj(hidden_states)
                
                q = F.silu(q_attn)
                k = F.silu(k_attn)
                v = F.silu(v_attn)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        
        q_attn, k_attn = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_attn_k_dim), (q_attn, k_attn))
        v_attn = rearrange(v_attn, '... (h d) -> ... h d', d=self.head_attn_v_dim)
        # TODO grouped QKV for attns...

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        if self.attn_qk_norm:
            q_attn, k_attn = self.q_norm(q_attn), self.k_norm(k_attn)

        if self.attn_apply_pos_encoding:
            # equivalent to cu_seqlens in `flash_attn`
            cu_seqlens = kwargs.get('cu_seqlens', None)

            seqlen_offset, max_seqlen = 0, q_len
            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                max_seqlen = q.shape[1] + seqlen_offset

                if attention_mask is not None:
                    # to deliminate the offsets of padding tokens
                    seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                    max_seqlen = q.shape[1] + max(seqlen_offset)

            if self.attn_max_position_embeddings is not None:
                max_seqlen = max(max_seqlen, self.attn_max_position_embeddings)
            q_attn, k_attn = self.rotary(q_attn, k_attn, seqlen_offset=seqlen_offset,
                                         max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        recurrent_state_gated_delta = last_state['recurrent_state_gated_delta'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state_gated_delta = nats_mixed_attn(
                q_attn=q_attn, k_attn=k_attn, v_attn=v_attn,
                q_lattn=q, k_lattn=k, v_lattn=v,
                initial_state_gated_delta=recurrent_state_gated_delta,
                g=g, beta=beta,
                nats_block_types=nats_op_types,
                n_nats_blocks=n_nats_blocks,
                scale_attn=k_attn.shape[-1] ** -0.5, scale_lattn=k.shape[-1] ** -0.5,
                cu_seqlens=cu_seqlens, cu_seqlens_nats=None,
                nats_block_size=self.nats_block_size,
                ops_for_incomplete_chunks=self.ops_for_incomplete_chunks,
                output_final_state_gated_delta=use_cache,
                compute_dnats_for_invalid_blocks_attn=self.compute_dnats_for_invalid_blocks_attn,
                compute_dnats_for_invalid_blocks_linear_att=self.compute_dnats_for_invalid_blocks_linear_att,
                incomplete_block_start_with_ht=self.incomplete_block_start_with_ht,
                use_g_for_attn=self.usg_for_attn,
                lattn_use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'fused_recurrent':
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            raise NotImplementedError
            #past_key_values.update(
            #    recurrent_state=recurrent_state,
            #    conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            #    layer_idx=self.layer_idx,
            #    offset=q_len
            #)

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


def test_mixed_attn():
    torch.manual_seed(0)
    device = torch.device('cuda')
    from nats.utils import check_fp16_dtype
    from torch.nn import functional as F

    dtype = torch.bfloat16 if check_fp16_dtype() == 'bfloat16' else torch.float16

    layer = NeuralAttentionSearchLinear(hidden_size=512, head_dim=128, num_heads=2, num_attn_heads=4,
                                        ops_for_incomplete_chunks='attn',
                                        compute_dnats_for_invalid_blocks_linear_att=False,
                                        compute_dnats_for_invalid_blocks_attn=False
                                        ).cuda()
    input_data = torch.randn((2, 512, 512)).cuda()
    out = layer(input_data)[0]

    loss = (out**2).sum()
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
