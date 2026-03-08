"""
RPPGTransformer — inference wrapper around QuantumInspiredRPPGTransformer (v8_filtered).

Architecture summary
--------------------
  RGB (B, T, 3)
  → SignalEmbedding (Linear → LN → GELU → Dropout → Linear → LN)
  → SinusoidalPositionalEncoding
  → 2 × TransformerEncoderLayer (pre-norm, DropPath)
       QuantumInspiredAttention (Sinkhorn doubly-stochastic, phase/entanglement/gate)
       + FFN (GELU)
  → head: LN → Linear(32, 32) → GELU → Dropout → Linear(32, 1)
  → BVP (B, T)

Config (v8_filtered)
--------------------
  embed_dim=32, num_heads=2, num_layers=2, dim_feedforward=64,
  max_seq_len=160, dropout=0.2, drop_path=0.1,
  sinkhorn_iter=20, sinkhorn_temperature=1.0 (learnable),
  head_hidden_dim=32
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import PHYSFORMER_WEIGHTS
from models.model_config import ModelConfig
from models.rppg_model import QuantumInspiredRPPGTransformer

logger = logging.getLogger(__name__)

_V8_CONFIG = ModelConfig()   # all defaults match v8_filtered


class RPPGTransformer:
    """
    Inference wrapper for the Quantum-Inspired rPPG Transformer.

    Parameters
    ----------
    weights_path : Path or str, optional
        Path to a .pth state_dict checkpoint.
        If None or file absent, the model runs with random weights
        (useful for unit tests / smoke testing).
    device : str
        Torch device string, e.g. "cpu" or "cuda".
    """

    def __init__(
        self,
        weights_path: Optional[Path] = PHYSFORMER_WEIGHTS,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = QuantumInspiredRPPGTransformer(config=_V8_CONFIG).to(self.device)
        self.model.eval()

        if weights_path is not None and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location=self.device)
            # Support both raw state_dict and training checkpoint formats
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state)
            logger.info("Loaded v8_filtered weights from %s", weights_path)
        else:
            logger.warning(
                "Weights not found at %s — using random init. "
                "Place physformer_lite.pth in src/models/weights/.",
                weights_path,
            )

    @torch.no_grad()
    def predict(self, rgb_series: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        rgb_series : np.ndarray, shape (T, 3), float32
            Mean-RGB time-series for one ROI region.
            Will be truncated to max_seq_len (160) if longer.

        Returns
        -------
        np.ndarray, shape (T,)
            Predicted BVP signal (z-score normalised).
        """
        if rgb_series.ndim != 2 or rgb_series.shape[1] != 3:
            raise ValueError(f"Expected shape (T, 3), got {rgb_series.shape}")

        # Truncate to model's max_seq_len
        max_len = _V8_CONFIG.max_seq_len
        if rgb_series.shape[0] > max_len:
            rgb_series = rgb_series[:max_len]

        # Normalise input per-channel
        mu = rgb_series.mean(axis=0, keepdims=True)
        sigma = rgb_series.std(axis=0, keepdims=True) + 1e-8
        x = (rgb_series - mu) / sigma  # (T, 3)

        tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)  # (1, T, 3)

        # Model returns (bvp, attention); we only need bvp
        bvp, _ = self.model(tensor, return_attention=False)  # (1, T)
        bvp_np = bvp.squeeze(0).cpu().numpy()                # (T,)

        # Z-score normalise output
        bvp_np = (bvp_np - bvp_np.mean()) / (bvp_np.std() + 1e-8)
        return bvp_np.astype(np.float32)
