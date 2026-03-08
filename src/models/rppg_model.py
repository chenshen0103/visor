"""Main Quantum-Inspired rPPG Transformer model."""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from models.model_config import ModelConfig
from models.embeddings import SignalEmbedding, SinusoidalPositionalEncoding
from models.transformer import TransformerEncoder
from models.attention.sinkhorn import verify_doubly_stochastic


class QuantumInspiredRPPGTransformer(nn.Module):
    """
    Quantum-Inspired rPPG Transformer for remote photoplethysmography.

    Architecture:
        RGB Input (B, T, 3)
            ↓
        Signal Embedding (B, T, D)
            ↓
        Positional Encoding (B, T, D)
            ↓
        N × Transformer Encoder Layers (with quantum-inspired attention)
            ↓
        Output Head (B, T, 1)
            ↓
        rPPG Signal (B, T)

    The quantum-inspired attention mechanism uses:
    - Phase rotation to simulate quantum gate operations
    - Entanglement simulation for non-local correlations
    - Sinkhorn normalization for doubly stochastic attention matrices
    """

    def __init__(
        self,
        config: ModelConfig = None,
        in_channels: int = None,
        embed_dim: int = None,
        num_heads: int = None,
        num_layers: int = None,
        dim_feedforward: int = None,
        dropout: float = None,
        drop_path: float = None,
        max_seq_len: int = None,
        sinkhorn_iter: int = None,
        sinkhorn_temperature: float = None,
        sinkhorn_convergence_threshold: float = None,
        sinkhorn_symmetric_final: bool = None,
    ):
        """
        Initialize the model.

        Can be initialized either with a ModelConfig object or individual parameters.
        If both are provided, individual parameters override config values.

        Args:
            config: Model configuration object
            in_channels: Number of input channels (RGB = 3)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            sinkhorn_iter: Number of Sinkhorn iterations
            sinkhorn_temperature: Initial Sinkhorn temperature
            sinkhorn_convergence_threshold: Early stopping threshold for DSM
            sinkhorn_symmetric_final: Apply symmetric final normalization
        """
        super().__init__()

        # Use config or defaults
        config = config or ModelConfig()

        # Allow parameter overrides
        self.in_channels = in_channels or config.in_channels
        self.embed_dim = embed_dim or config.embed_dim
        self.num_heads = num_heads or config.num_heads
        self.num_layers = num_layers or config.num_layers
        self.dim_feedforward = dim_feedforward or config.dim_feedforward
        self.dropout = dropout if dropout is not None else config.dropout
        self.drop_path = drop_path if drop_path is not None else getattr(config, 'drop_path', 0.0)
        self.max_seq_len = max_seq_len or config.max_seq_len
        self.sinkhorn_iter = sinkhorn_iter or config.sinkhorn_iter
        self.sinkhorn_temperature = sinkhorn_temperature or config.sinkhorn_temperature
        self.sinkhorn_convergence_threshold = sinkhorn_convergence_threshold or getattr(config, 'sinkhorn_convergence_threshold', 1e-3)
        self.sinkhorn_symmetric_final = sinkhorn_symmetric_final if sinkhorn_symmetric_final is not None else getattr(config, 'sinkhorn_symmetric_final', True)

        # Signal embedding
        self.embedding = SignalEmbedding(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=self.embed_dim,
            max_len=self.max_seq_len,
            dropout=self.dropout,
        )

        # Transformer encoder with quantum-inspired attention
        self.encoder = TransformerEncoder(
            dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            drop_path=self.drop_path,
            sinkhorn_iter=self.sinkhorn_iter,
            sinkhorn_temperature=self.sinkhorn_temperature,
            sinkhorn_convergence_threshold=self.sinkhorn_convergence_threshold,
            sinkhorn_symmetric_final=self.sinkhorn_symmetric_final,
            use_quantum=True,
        )

        # Output head: project to rPPG signal
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Args:
            x: Input RGB signal (B, T, 3)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (rppg_signal, attention_weights)
            - rppg_signal: Predicted rPPG signal (B, T)
            - attention_weights: List of attention matrices if requested
        """
        # Embed input signals
        x = self.embedding(x)  # (B, T, D)

        # Add positional encoding
        x = self.pos_encoding(x)  # (B, T, D)

        # Pass through transformer encoder
        x, attention = self.encoder(x, return_all_attention=return_attention)

        # Project to rPPG signal
        x = self.head(x)  # (B, T, 1)
        x = x.squeeze(-1)  # (B, T)

        return x, attention

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-only forward pass.

        Args:
            x: Input RGB signal (B, T, 3)

        Returns:
            Predicted rPPG signal (B, T)
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(x, return_attention=False)
        return output

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps for visualization.

        Args:
            x: Input RGB signal (B, T, 3)

        Returns:
            List of attention tensors, one per layer (B, H, T, T)
        """
        self.eval()
        with torch.no_grad():
            _, attention = self.forward(x, return_attention=True)
        return attention

    def verify_doubly_stochastic(
        self,
        x: torch.Tensor,
        tolerance: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Verify that attention matrices are doubly stochastic.

        Args:
            x: Input RGB signal (B, T, 3)
            tolerance: Tolerance for verification

        Returns:
            Dictionary with verification results per layer
        """
        attention_maps = self.get_attention_maps(x)

        results = {}
        for i, attn in enumerate(attention_maps):
            is_ds, row_err, col_err = verify_doubly_stochastic(attn, tolerance)
            results[f"layer_{i}"] = {
                "is_doubly_stochastic": is_ds,
                "row_error": row_err,
                "col_error": col_err,
            }

        return results

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model architecture information.

        Returns:
            Dictionary with model information
        """
        return {
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "sinkhorn_iter": self.sinkhorn_iter,
            "parameters": self.count_parameters(),
        }


class ClassicalRPPGTransformer(QuantumInspiredRPPGTransformer):
    """
    Classical rPPG Transformer with standard softmax attention.

    This is a baseline model for comparison with the quantum-inspired version.
    Uses the same architecture but with classical multi-head attention.
    """

    def __init__(self, config: ModelConfig = None, **kwargs):
        # First initialize parent to get config
        super().__init__(config=config, **kwargs)

        # Replace encoder with classical version
        self.encoder = TransformerEncoder(
            dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            use_quantum=False,  # Use classical attention
        )

    def verify_doubly_stochastic(self, x, tolerance=1e-3):
        """Classical attention uses softmax, not doubly stochastic."""
        return {"note": "Classical model uses softmax, not doubly stochastic attention"}
