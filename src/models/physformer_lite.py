"""
PhysFormerLite — lightweight Transformer for rPPG BVP estimation.

Input : (B, T, 3)   — mean RGB time-series per spatial region
Output: (B, T)      — predicted BVP (blood-volume-pulse) signal

Architecture: 4-layer Transformer encoder with a linear projection head.
Inspired by PhysFormer (Yu et al., 2022) but stripped to fit inference on CPU.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from config import (
    PHYSFORMER_D_MODEL,
    PHYSFORMER_N_HEADS,
    PHYSFORMER_N_LAYERS,
    PHYSFORMER_DFF,
    PHYSFORMER_DROPOUT,
    PHYSFORMER_MAX_SEQ_LEN,
)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PhysFormerLite(nn.Module):
    """
    4-layer Transformer encoder that maps a raw RGB time-series to a BVP signal.

    Parameters
    ----------
    d_model : int
        Internal feature dimension (default 64).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of encoder layers (default 4).
    d_ff : int
        Feed-forward hidden dimension (default 128).
    dropout : float
        Dropout probability (default 0.1).
    max_seq_len : int
        Maximum sequence length (default 512).
    """

    def __init__(
        self,
        d_model: int = PHYSFORMER_D_MODEL,
        n_heads: int = PHYSFORMER_N_HEADS,
        n_layers: int = PHYSFORMER_N_LAYERS,
        d_ff: int = PHYSFORMER_DFF,
        dropout: float = PHYSFORMER_DROPOUT,
        max_seq_len: int = PHYSFORMER_MAX_SEQ_LEN,
    ) -> None:
        super().__init__()

        # Project 3-channel RGB input to d_model
        self.input_proj = nn.Linear(3, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # (B, T, d_model)
            norm_first=True,   # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project to scalar BVP per time-step
        self.output_proj = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, 3)
            Mean RGB values per frame per region.
        src_key_padding_mask : BoolTensor, shape (B, T), optional
            True positions are ignored (padding).

        Returns
        -------
        Tensor, shape (B, T)
            Predicted BVP signal.
        """
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_enc(x)             # (B, T, d_model)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)
        x = self.output_proj(x)         # (B, T, 1)
        return x.squeeze(-1)            # (B, T)
