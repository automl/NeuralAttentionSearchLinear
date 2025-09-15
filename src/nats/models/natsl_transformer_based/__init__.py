
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nats.models.natsl_transformer_based.configuration_natsl_transformer_based import NeuralAttentionSearchLinearTransformerBasedConfig
from nats.models.natsl_transformer_based.modeling_natsl_transformer_based import (
    NeuralAttentionSearchLinearTransformerBasedForCausalLM,
    NeuralAttentionSearchLinearTransformerBasedModel
)

AutoConfig.register(NeuralAttentionSearchLinearTransformerBasedConfig.model_type, NeuralAttentionSearchLinearTransformerBasedConfig, exist_ok=True)
AutoModel.register(NeuralAttentionSearchLinearTransformerBasedConfig, NeuralAttentionSearchLinearTransformerBasedModel, exist_ok=True)
AutoModelForCausalLM.register(NeuralAttentionSearchLinearTransformerBasedConfig, NeuralAttentionSearchLinearTransformerBasedForCausalLM, exist_ok=True)

__all__ = ['NeuralAttentionSearchLinearTransformerBasedConfig',
           'NeuralAttentionSearchLinearTransformerBasedModel',
           'NeuralAttentionSearchLinearTransformerBasedForCausalLM']