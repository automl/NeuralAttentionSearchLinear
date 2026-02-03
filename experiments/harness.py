# -*- coding: utf-8 -*-

from __future__ import annotations

import nats  # noqa
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model('nats')
class NAtSLMWrapper(HFLM):
    def __init__(self, **kwargs) -> NAtSLMWrapper:

        # TODO: provide options for doing inference with different kernels

        super().__init__(**kwargs)


if __name__ == "__main__":
    cli_evaluate()
