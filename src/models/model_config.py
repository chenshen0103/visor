"""
ModelConfig — v8_filtered architecture parameters.

This replaces the original training project's config/default.py so that
rppg_model.py can be imported without the original package structure.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Input / sequence
    in_channels: int = 3
    max_seq_len: int = 160

    # Transformer
    embed_dim: int = 32
    num_heads: int = 2
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.2
    drop_path: float = 0.1

    # Sinkhorn attention
    sinkhorn_iter: int = 20
    sinkhorn_temperature: float = 1.0
    learnable_temperature: bool = True
    sinkhorn_convergence_threshold: float = 0.001
    sinkhorn_symmetric_final: bool = True

    # Quantum-inspired init
    phase_init_scale: float = 0.1
    entanglement_init: float = 0.3
    gate_init: float = 0.5

    # Output head
    head_hidden_dim: int = 32
