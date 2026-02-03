import json
import os
from setuptools import setup, find_packages


setup(
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'torch==2.7.1',
        'triton==3.3.1',
        'transformers==4.56.1',
        'wandb==0.21.1',
        'einops==0.8.1',
        'flash-attn==2.8.3',
        'flash-linear-attention==0.4.0',
        'huggingface-hub==0.36.0',
        'mamba-ssm==2.2.5',
        'numpy==2.3.2',
        'pandas==2.3.1',
        'tokenizers==0.22.2',
        'torchtitan==0.0.2',
        'datasets==3.3.0'
    ],
)