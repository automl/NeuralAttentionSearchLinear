# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class NeuralAttentionSearchLinearTransformerBasedConfig(PretrainedConfig):
    model_type = 'neural_attention_search_linear_transformer_based'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
            self,
            attn_mode: str = "chunk",
            hidden_size: int = 2048,
            num_heads: int = 32,
            num_lattn_heads: int = 8,
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
            lattns_use_silu: bool = False,
            outputs_are_wighted: bool = True,
            compute_dnats_for_invalid_blocks_attn: bool = False,
            compute_dnats_for_invalid_blocks_linear_att: bool = False,
            incomplete_block_start_with_ht: bool = True,
            attn_apply_pos_encoding: bool = True,
            rope_theta: Optional[float] = 10000.,
            max_position_embeddings: Optional[int] = None,
            norm_eps: float = 1e-6,
            num_hidden_layers: int = 21,
            hidden_ratio: Optional[int] = 4,
            intermediate_size: Optional[int] = None,
            hidden_act: str = "swish",
            initializer_range: float = 0.02,
            elementwise_affine: Optional[bool] = True,
            use_cache: bool = True,
            pad_token_id: Optional[int] = None,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            tie_word_embeddings: bool = False,
            fuse_norm: bool = True,
            fuse_swiglu: bool = True,
            fuse_cross_entropy: bool = True,
            use_l2warp: bool = False,
            vocab_size: int = 32000,
            **kwargs
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_lattn_heads = num_lattn_heads
        self.num_kv_heads = num_kv_heads
        self.num_nats_head = num_nats_head

        self.qk_norm = qk_norm
        self.qkv_bias = qkv_bias
        self.window_size = window_size
        self.mode=mode

        self.max_position_embeddings = max_position_embeddings

        self.n_ops = n_ops
        self.outputs_are_wighted = outputs_are_wighted
        self.nats_block_size = nats_block_size
        self.nats_block_agg_type = nats_block_agg_type
        self.nats_sample_strategy = nats_sample_strategy
        self.ops_for_incomplete_chunks = ops_for_incomplete_chunks

        self.compute_dnats_for_invalid_blocks_attn = compute_dnats_for_invalid_blocks_attn
        self.compute_dnats_for_invalid_blocks_linear_att = compute_dnats_for_invalid_blocks_linear_att
        self.incomplete_block_start_with_ht = incomplete_block_start_with_ht

        self.lattns_use_silu = lattns_use_silu

        self.attn_apply_pos_encoding = attn_apply_pos_encoding
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.allow_neg_eigval = allow_neg_eigval

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
