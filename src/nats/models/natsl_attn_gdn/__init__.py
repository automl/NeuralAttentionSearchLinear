# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

#from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
#from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetModel

from nats.models.natsl_attn_gdn.configuration_natsl_attn_gdn import NeuralAttentionSearchLineraAttnGDNConfig
from nats.models.natsl_attn_gdn.modeling_natsl_attn_gdn import (
    NeuralAttentionSearchLinearAttnGDNForCausalLM,
    NeuralAttentionSearchLinearAttnGDNModel
)

AutoConfig.register(NeuralAttentionSearchLineraAttnGDNConfig.model_type,
                    NeuralAttentionSearchLineraAttnGDNConfig,
                    exist_ok=True
                    )
AutoModel.register(NeuralAttentionSearchLineraAttnGDNConfig,
                   NeuralAttentionSearchLinearAttnGDNModel,
                   exist_ok=True
                   )
AutoModelForCausalLM.register(
    NeuralAttentionSearchLineraAttnGDNConfig,
    NeuralAttentionSearchLinearAttnGDNForCausalLM,
    exist_ok=True
)

__all__ = [
    'NeuralAttentionSearchLineraAttnGDNConfig',
    'NeuralAttentionSearchLinearAttnGDNForCausalLM',
    'NeuralAttentionSearchLinearAttnGDNModel'
]
