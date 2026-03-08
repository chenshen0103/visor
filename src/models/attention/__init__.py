"""Attention mechanisms module."""

from .sinkhorn import LogSpaceSinkhorn, verify_doubly_stochastic
from .quantum_inspired import QuantumInspiredAttention

__all__ = [
    "LogSpaceSinkhorn",
    "verify_doubly_stochastic",
    "QuantumInspiredAttention",
]
