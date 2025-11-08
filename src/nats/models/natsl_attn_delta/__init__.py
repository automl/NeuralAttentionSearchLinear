# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


from nats.models.natsl_attn_delta.configuration_natsl_attn_delta import NeuralAttentionSearchLineraAttnDeltaConfig
from nats.models.natsl_attn_delta.modeling_natsl_attn_delta import (
    NeuralAttentionSearchLinearAttnDeltaForCausalLM,
    NeuralAttentionSearchLinearAttnDeltaModel
)

AutoConfig.register(NeuralAttentionSearchLineraAttnDeltaConfig.model_type,
                    NeuralAttentionSearchLineraAttnDeltaConfig,
                    exist_ok=True
                    )
AutoModel.register(NeuralAttentionSearchLineraAttnDeltaConfig,
                   NeuralAttentionSearchLinearAttnDeltaModel,
                   exist_ok=True
                   )
AutoModelForCausalLM.register(
    NeuralAttentionSearchLineraAttnDeltaConfig,
    NeuralAttentionSearchLinearAttnDeltaForCausalLM,
    exist_ok=True
)

__all__ = [
    'NeuralAttentionSearchLineraAttnDeltaConfig',
    'NeuralAttentionSearchLinearAttnDeltaForCausalLM',
    'NeuralAttentionSearchLinearAttnDeltaModel'
]
