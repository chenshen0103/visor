"""Transformer encoder layers."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .attention.quantum_inspired import QuantumInspiredAttention, ClassicalMultiHeadAttention


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Randomly drops entire residual branches during training.
    This is a regularization technique that helps prevent overfitting.

    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """

    def __init__(self, drop_prob: float = 0.0):
        """
        Initialize DropPath.

        Args:
            drop_prob: Probability of dropping the path (0 = no drop, 1 = always drop)
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Work with any tensor shape
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer with quantum-inspired attention.

    Uses pre-normalization (LayerNorm before attention/FFN) which provides
    more stable training compared to post-normalization.

    Architecture:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> + -> output
        |___________________________|  |______________________|
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        sinkhorn_iter: int = 20,
        sinkhorn_temperature: float = 1.0,
        sinkhorn_convergence_threshold: float = 1e-3,
        sinkhorn_symmetric_final: bool = True,
        use_quantum: bool = True,
    ):
        """
        Initialize transformer layer.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout probability
            drop_path: Drop path (stochastic depth) probability
            sinkhorn_iter: Number of Sinkhorn iterations
            sinkhorn_temperature: Initial Sinkhorn temperature
            sinkhorn_convergence_threshold: Early stopping threshold for DSM
            sinkhorn_symmetric_final: Apply symmetric final normalization
            use_quantum: Whether to use quantum-inspired attention
        """
        super().__init__()

        self.dim = dim
        self.use_quantum = use_quantum

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # DropPath for stochastic depth regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Attention
        if use_quantum:
            self.attn = QuantumInspiredAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                sinkhorn_iter=sinkhorn_iter,
                sinkhorn_temperature=sinkhorn_temperature,
                sinkhorn_convergence_threshold=sinkhorn_convergence_threshold,
                sinkhorn_symmetric_final=sinkhorn_symmetric_final,
            )
        else:
            self.attn = ClassicalMultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize FFN weights."""
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        """
        # Pre-norm attention with residual and drop path
        attn_output, attn_weights = self.attn(
            self.norm1(x), return_attention=return_attention
        )
        x = x + self.drop_path(attn_output)

        # Pre-norm FFN with residual and drop path
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers.

    Provides a convenient wrapper for multiple transformer layers.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        sinkhorn_iter: int = 20,
        sinkhorn_temperature: float = 1.0,
        sinkhorn_convergence_threshold: float = 1e-3,
        sinkhorn_symmetric_final: bool = True,
        use_quantum: bool = True,
    ):
        """
        Initialize transformer encoder.

        Args:
            dim: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout probability
            drop_path: Maximum drop path rate (uses linear scaling per layer)
            sinkhorn_iter: Number of Sinkhorn iterations
            sinkhorn_temperature: Initial Sinkhorn temperature
            sinkhorn_convergence_threshold: Early stopping threshold for DSM
            sinkhorn_symmetric_final: Apply symmetric final normalization
            use_quantum: Whether to use quantum-inspired attention
        """
        super().__init__()

        # Stochastic depth: linear decay from 0 to drop_path
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, num_layers)]

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                drop_path=drop_path_rates[i],
                sinkhorn_iter=sinkhorn_iter,
                sinkhorn_temperature=sinkhorn_temperature,
                sinkhorn_convergence_threshold=sinkhorn_convergence_threshold,
                sinkhorn_symmetric_final=sinkhorn_symmetric_final,
                use_quantum=use_quantum,
            )
            for i in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        return_all_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor (B, T, D)
            return_all_attention: Whether to return attention from all layers

        Returns:
            Tuple of (output, attention_list)
        """
        all_attention = [] if return_all_attention else None

        for layer in self.layers:
            x, attn = layer(x, return_attention=return_all_attention)
            if return_all_attention:
                all_attention.append(attn)

        x = self.norm(x)

        return x, all_attention
