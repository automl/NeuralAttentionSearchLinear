# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


from nats.models.natsl_attn_mamba2.configuration_nats_mamba2 import NATSMamba2Config
from nats.models.natsl_attn_mamba2.modeling_nats_mamba2 import NAtSMamba2ForCausalLM, NAtSMamba2Model

AutoConfig.register(NATSMamba2Config.model_type, NATSMamba2Config, exist_ok=True)
AutoModel.register(NATSMamba2Config, NAtSMamba2Model, exist_ok=True)
AutoModelForCausalLM.register(NATSMamba2Config, NAtSMamba2ForCausalLM, exist_ok=True)


__all__ = ['NATSMamba2Config', 'NAtSMamba2ForCausalLM', 'NAtSMamba2Model']
