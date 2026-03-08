"""
Quantum-Inspired Attention Mechanism.

This module implements a quantum-inspired attention mechanism that replaces
the traditional softmax with doubly stochastic normalization (Sinkhorn).

Key innovations:
1. Phase Rotation: Simulates quantum phase gates using rotation matrices
2. Entanglement Simulation: Models non-local correlations via self-interaction
3. Gate Mechanism: Learnable blending of classical and quantum components
4. Doubly Stochastic Output: Sinkhorn normalization ensures row/column sums = 1

The doubly stochastic constraint provides more uniform attention distribution
compared to softmax, which can be too peaked (winner-take-all) or too flat.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sinkhorn import LogSpaceSinkhorn


class QuantumInspiredAttention(nn.Module):
    """
    Quantum-Inspired Multi-Head Attention with Doubly Stochastic Normalization.

    Mathematical motivation:
    - In quantum mechanics, unitary transformations preserve probabilities
    - Doubly stochastic matrices arise from quantum channels and measurements
    - Phase rotations simulate unitary quantum gates
    - Entanglement creates non-local correlations between positions

    This provides an inductive bias for attention that:
    - Distributes attention more uniformly (no single dominant position)
    - Preserves information flow (both source and target get "fair" attention)
    - Learns to mix classical and quantum-like behaviors adaptively
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        dropout: float = 0.1,
        sinkhorn_iter: int = 5,
        sinkhorn_temperature: float = 1.0,
        learnable_temperature: bool = True,
        sinkhorn_convergence_threshold: float = 1e-3,
        sinkhorn_symmetric_final: bool = True,
        phase_init_scale: float = 0.1,
        entanglement_init: float = 0.3,
        gate_init: float = 0.5,
    ):
        """
        Initialize quantum-inspired attention.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            dropout: Dropout probability
            sinkhorn_iter: Number of Sinkhorn iterations
            sinkhorn_temperature: Initial temperature for Sinkhorn
            learnable_temperature: Whether temperature is learnable
            sinkhorn_convergence_threshold: Early stopping threshold for DSM
            sinkhorn_symmetric_final: Apply symmetric final normalization
            phase_init_scale: Initialization scale for phase rotation
            entanglement_init: Initial entanglement strength
            gate_init: Initial gate value (0 = classical, 1 = quantum)
        """
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Quantum-inspired learnable parameters
        # Phase rotation: one angle per head
        self.phase_rotation = nn.Parameter(
            torch.randn(num_heads, 1, 1) * phase_init_scale
        )

        # Entanglement strength: controls self-interaction magnitude
        self.entanglement_strength = nn.Parameter(
            torch.ones(num_heads) * entanglement_init
        )

        # Gate: blends classical and quantum attention
        # sigmoid(gate) = 0 -> pure classical, 1 -> pure quantum
        self.quantum_gate = nn.Parameter(
            torch.ones(num_heads) * gate_init
        )

        # Sinkhorn normalizer
        self.sinkhorn = LogSpaceSinkhorn(
            num_iterations=sinkhorn_iter,
            temperature=sinkhorn_temperature,
            learnable_temperature=learnable_temperature,
            convergence_threshold=sinkhorn_convergence_threshold,
            use_symmetric_final=sinkhorn_symmetric_final,
        )

    def quantum_phase_rotation(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply quantum-inspired phase rotation.

        Simulates a unitary rotation gate by applying a rotation matrix
        in the real-valued domain. This creates interference patterns
        similar to quantum phase interactions.

        Args:
            x: Input tensor (B, H, T, T) - attention scores

        Returns:
            Phase-rotated scores (B, H, T, T)
        """
        # Get rotation angle per head
        theta = self.phase_rotation  # (H, 1, 1)

        # Apply 2D rotation in the attention score space
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Rotate the attention pattern
        # This creates constructive/destructive interference effects
        x_rotated = cos_theta * x - sin_theta * x.transpose(-2, -1)
        x_rotated = x_rotated + sin_theta * x + cos_theta * x.transpose(-2, -1)
        x_rotated = x_rotated / 2  # Normalize

        return x_rotated

    def quantum_entanglement(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate quantum entanglement via self-interaction.

        In quantum systems, entangled particles have correlated properties.
        We simulate this by computing pairwise interactions that create
        non-local correlations between positions.

        Args:
            x: Input tensor (B, H, T, D) - query or key representations

        Returns:
            Entanglement contribution (B, H, T, T)
        """
        # Self-interaction: x @ x^T creates pairwise correlations
        # This is similar to how entanglement creates non-local correlations
        interaction = torch.einsum("bhid,bhjd->bhij", x, x)

        # Scale by entanglement strength (per head)
        strength = self.entanglement_strength.view(1, -1, 1, 1)  # (1, H, 1, 1)
        entangled = interaction * strength

        return entangled

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, D)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
            - output: (B, T, D)
            - attention_weights: (B, H, T, T) if return_attention else None
        """
        B, T, D = x.shape

        # Compute QKV
        qkv = self.qkv(x)  # (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D')
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, T, D')

        # Classical attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Quantum-inspired transformations
        # 1. Phase rotation
        rotated = self.quantum_phase_rotation(attn_scores)

        # 2. Entanglement (computed from queries)
        entangled = self.quantum_entanglement(q)

        # 3. Gate mechanism: blend classical and quantum
        gate = torch.sigmoid(self.quantum_gate).view(1, -1, 1, 1)  # (1, H, 1, 1)
        quantum_scores = rotated + 0.1 * entangled  # Combine quantum effects
        mixed_scores = (1 - gate) * attn_scores + gate * quantum_scores

        # 4. Apply Sinkhorn normalization for doubly stochastic output
        # First, ensure scores are in a reasonable range for log-space
        log_scores = mixed_scores  # Already in a suitable range
        attn_weights = self.sinkhorn(log_scores)  # (B, H, T, T)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D')
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)  # (B, T, D)

        # Final projection
        output = self.proj(attn_output)
        output = self.dropout(output)

        if return_attention:
            return output, attn_weights
        return output, None

    def get_attention_stats(self) -> dict:
        """Get statistics about learned quantum parameters."""
        return {
            "phase_rotation_mean": self.phase_rotation.mean().item(),
            "phase_rotation_std": self.phase_rotation.std().item(),
            "entanglement_strength": self.entanglement_strength.mean().item(),
            "gate_values": torch.sigmoid(self.quantum_gate).tolist(),
            "sinkhorn_temperature": self.sinkhorn.tau.item(),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, sinkhorn_iter={self.sinkhorn.num_iterations}"
        )


class ClassicalMultiHeadAttention(nn.Module):
    """
    Standard multi-head attention for comparison.

    This provides a baseline using traditional softmax normalization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # Standard softmax
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)

        output = self.proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None
