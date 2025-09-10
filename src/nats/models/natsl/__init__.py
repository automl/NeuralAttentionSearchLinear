# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

#from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
#from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetModel

from nats.models.natsl.configuration_natsl import NeuralAttentionSearchLineraConfig
from nats.models.natsl.modeling_natsl import NeuralAttentionSearchLinearForCausalLM, NeuralAttentionSearchLinearModel

AutoConfig.register(NeuralAttentionSearchLineraConfig.model_type, NeuralAttentionSearchLineraConfig, exist_ok=True)
AutoModel.register(NeuralAttentionSearchLineraConfig, NeuralAttentionSearchLinearModel, exist_ok=True)
AutoModelForCausalLM.register(NeuralAttentionSearchLineraConfig, NeuralAttentionSearchLinearForCausalLM, exist_ok=True)

__all__ = ['NeuralAttentionSearchLineraConfig', 'NeuralAttentionSearchLineraConfig', 'NeuralAttentionSearchLinearModel']
