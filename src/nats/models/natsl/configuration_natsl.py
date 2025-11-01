# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class NeuralAttentionSearchLineraConfig(PretrainedConfig):
    model_type = 'neural_attention_search_linear'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
            self,
            attn_mode: str = "chunk",
            hidden_size: int = 2048,
            expand_v: float = 2.0,
            use_gate: bool = True,
            use_short_conv: bool = True,
            allow_neg_eigval: bool = False,
            conv_size: int = 4,
            conv_bias: bool = False,
            head_dim: int = 256,
            num_heads: int = 6,
            num_attn_heads: int = 12,
            num_v_heads: Optional[int] = None,
            n_ops: int = 2,
            num_nats_head: Optional[int] = None,
            nats_block_size: int = 64,
            nats_block_agg_type: str = 'mean',
            nats_sample_strategy: str = 'argmax',
            ops_for_incomplete_chunks: str = 'all',
            max_position_embeddings: int = 2048,
            hidden_ratio: Optional[int] = 4,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "swish",
            outputs_are_wighted: bool = True,
            attn_qk_norm: bool = False,
            attn_with_short_conv: bool = False,
            compute_dnats_for_invalid_blocks_attn: bool = False,
            compute_dnats_for_invalid_blocks_linear_att: bool = False,
            incomplete_block_start_with_ht: bool = True,
            attn_apply_pos_encoding: bool = True,
            attn_rope_theta: Optional[float] = 10000.,
            attn_max_position_embeddings: Optional[int] = None,
            num_hidden_layers: int = 21,
            norm_eps: float = 1e-6,
            attn: Optional[Dict] = None,
            use_cache: bool = True,
            pad_token_id: Optional[int] = None,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            tie_word_embeddings: bool = False,
            initializer_range: float = 0.02,
            fuse_norm: bool = True,
            fuse_swiglu: bool = True,
            fuse_cross_entropy: bool = True,
            use_l2warp: bool = False,
            vocab_size: int = 32000,
            gdn_layers: list[int] | None = None,
            **kwargs
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_attn_heads = num_attn_heads
        self.num_v_heads = num_v_heads
        self.max_position_embeddings = max_position_embeddings

        self.n_ops = n_ops
        self.num_nats_head = num_nats_head
        self.outputs_are_wighted = outputs_are_wighted
        self.attn_qk_norm = attn_qk_norm
        self.nats_block_size = nats_block_size
        self.nats_block_agg_type = nats_block_agg_type
        self.nats_sample_strategy = nats_sample_strategy
        self.ops_for_incomplete_chunks = ops_for_incomplete_chunks

        self.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        self.compute_dnats_for_invalid_blocks_linear_att = compute_dnats_for_invalid_blocks_linear_att
        self.incomplete_block_start_with_ht = incomplete_block_start_with_ht
        self.attn_with_short_conv = attn_with_short_conv
        self.attn_apply_pos_encoding = attn_apply_pos_encoding
        self.attn_rope_theta = attn_rope_theta
        self.attn_max_position_embeddings = attn_max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.allow_neg_eigval = allow_neg_eigval

        if gdn_layers is None:
            gdn_layers = []
        self.gdn_layer = gdn_layers

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
