"""Signal embeddings and positional encoding."""

import math
import torch
import torch.nn as nn


class SignalEmbedding(nn.Module):
    """
    Project RGB signals to embedding dimension.

    Takes raw 3-channel RGB signals and projects them into a higher-dimensional
    embedding space suitable for transformer processing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize signal embedding.

        Args:
            in_channels: Number of input channels (3 for RGB)
            embed_dim: Embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.projection = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C) where C = in_channels

        Returns:
            Embedded tensor (B, T, embed_dim)
        """
        return self.projection(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Adds position information to embeddings using sine and cosine functions
    at different frequencies. This allows the model to learn position-dependent
    patterns.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension (must match embedding dimension)
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = self._create_pe(d_model, max_len)
        self.register_buffer("pe", pe)

    def _create_pe(
        self,
        d_model: int,
        max_len: int,
    ) -> torch.Tensor:
        """Create positional encoding matrix."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term for even/odd positions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Position-encoded tensor (B, T, D)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.

    Alternative to sinusoidal encoding where position embeddings are learned
    during training. Can sometimes perform better for specific tasks.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize learnable positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Learnable position embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (B, T, D)

        Returns:
            Position-encoded tensor (B, T, D)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TemporalConvEmbedding(nn.Module):
    """
    Temporal convolutional embedding.

    Uses 1D convolutions to capture local temporal patterns before
    feeding into the transformer. This can help with learning
    short-term dependencies.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        kernel_sizes: tuple = (3, 5, 7),
        dropout: float = 0.1,
    ):
        """
        Initialize temporal conv embedding.

        Args:
            in_channels: Number of input channels
            embed_dim: Output embedding dimension
            kernel_sizes: Tuple of kernel sizes for multi-scale convolutions
            dropout: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Multi-scale 1D convolutions
        self.convs = nn.ModuleList()
        hidden_per_conv = embed_dim // len(kernel_sizes)

        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    hidden_per_conv,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.BatchNorm1d(hidden_per_conv),
                nn.GELU(),
            )
            self.convs.append(conv)

        # Final projection
        total_hidden = hidden_per_conv * len(kernel_sizes)
        self.proj = nn.Sequential(
            nn.Linear(total_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C)

        Returns:
            Embedded tensor (B, T, embed_dim)
        """
        # Transpose for conv: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))

        # Concatenate: (B, hidden_total, T)
        x = torch.cat(conv_outputs, dim=1)

        # Transpose back: (B, hidden_total, T) -> (B, T, hidden_total)
        x = x.transpose(1, 2)

        # Final projection
        x = self.proj(x)

        return x
