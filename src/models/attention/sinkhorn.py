"""Log-space Sinkhorn normalization for doubly stochastic matrices."""

from typing import Tuple
import torch
import torch.nn as nn


class LogSpaceSinkhorn(nn.Module):
    """
    Log-space Sinkhorn-Knopp algorithm for doubly stochastic normalization.

    Converts arbitrary scores into a doubly stochastic matrix where:
    - All row sums = 1
    - All column sums = 1
    - All values >= 0

    This provides more uniform attention distribution compared to softmax,
    which can become too peaked or too flat.

    The log-space formulation is numerically stable for attention matrices.
    """

    def __init__(
        self,
        num_iterations: int = 5,
        temperature: float = 1.0,
        learnable_temperature: bool = True,
        eps: float = 1e-6,
        convergence_threshold: float = 1e-3,
        use_symmetric_final: bool = True,
    ):
        """
        Initialize Sinkhorn normalizer.

        Args:
            num_iterations: Number of Sinkhorn iterations
            temperature: Temperature parameter (lower = sharper distribution)
            learnable_temperature: Whether temperature is learnable
            eps: Small constant for numerical stability
            convergence_threshold: Early stopping threshold for row/col error
            use_symmetric_final: Apply final symmetric normalization for better DSM
        """
        super().__init__()

        self.num_iterations = num_iterations
        self.eps = eps
        self.convergence_threshold = convergence_threshold
        self.use_symmetric_final = use_symmetric_final

        if learnable_temperature:
            self.tau = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("tau", torch.tensor(temperature))

    def forward(self, log_scores: torch.Tensor) -> torch.Tensor:
        """
        Convert log-scores to doubly stochastic matrix.

        Args:
            log_scores: Log attention scores (B, H, T, T)

        Returns:
            Doubly stochastic matrix (B, H, T, T) where rows and columns sum to 1
        """
        # Apply temperature scaling in log space
        tau = self.tau.clamp(min=0.1)  # Prevent division by zero
        log_scores = log_scores / tau

        # Sinkhorn iterations in log space
        for i in range(self.num_iterations):
            # Row normalization: subtract log of row sums
            log_scores = log_scores - torch.logsumexp(
                log_scores, dim=-1, keepdim=True
            )
            # Column normalization: subtract log of column sums
            log_scores = log_scores - torch.logsumexp(
                log_scores, dim=-2, keepdim=True
            )

            # Early convergence check (every 5 iterations to reduce overhead)
            if i > 0 and i % 5 == 0 and not self.training:
                with torch.no_grad():
                    dsm_check = torch.exp(log_scores)
                    row_error = (dsm_check.sum(dim=-1) - 1.0).abs().max()
                    col_error = (dsm_check.sum(dim=-2) - 1.0).abs().max()
                    if row_error < self.convergence_threshold and col_error < self.convergence_threshold:
                        break

        # Final symmetric normalization to balance row and column errors
        if self.use_symmetric_final:
            # One more row normalization to balance the final column normalization
            log_scores = log_scores - torch.logsumexp(
                log_scores, dim=-1, keepdim=True
            )
            # Average with column-normalized version for symmetry
            log_scores_col = log_scores - torch.logsumexp(
                log_scores, dim=-2, keepdim=True
            )
            # Weighted average in log space (geometric mean)
            log_scores = 0.5 * (log_scores + log_scores_col)

        # Convert back from log space
        dsm = torch.exp(log_scores)

        return dsm

    def forward_with_diagnostics(
        self, log_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with diagnostic information.

        Args:
            log_scores: Log attention scores (B, H, T, T)

        Returns:
            Tuple of (dsm, diagnostics_dict)
        """
        dsm = self.forward(log_scores)

        # Compute diagnostics
        row_sums = dsm.sum(dim=-1)
        col_sums = dsm.sum(dim=-2)

        diagnostics = {
            "row_sum_mean": row_sums.mean().item(),
            "row_sum_std": row_sums.std().item(),
            "col_sum_mean": col_sums.mean().item(),
            "col_sum_std": col_sums.std().item(),
            "temperature": self.tau.item(),
            "max_value": dsm.max().item(),
            "min_value": dsm.min().item(),
        }

        return dsm, diagnostics

    def extra_repr(self) -> str:
        return (
            f"num_iterations={self.num_iterations}, "
            f"learnable_temp={isinstance(self.tau, nn.Parameter)}, "
            f"symmetric_final={self.use_symmetric_final}"
        )


def verify_doubly_stochastic(
    attn: torch.Tensor,
    tolerance: float = 1e-3,
) -> Tuple[bool, float, float]:
    """
    Verify that a matrix is doubly stochastic.

    Args:
        attn: Attention matrix (B, H, T, T) or (T, T)
        tolerance: Tolerance for sum verification

    Returns:
        Tuple of (is_doubly_stochastic, row_error, col_error)
    """
    row_sums = attn.sum(dim=-1)
    col_sums = attn.sum(dim=-2)

    row_error = (row_sums - 1.0).abs().max().item()
    col_error = (col_sums - 1.0).abs().max().item()

    is_ds = (row_error < tolerance) and (col_error < tolerance)

    return is_ds, row_error, col_error


class GumbelSinkhorn(nn.Module):
    """
    Gumbel-Sinkhorn for differentiable permutation learning.

    This variant adds Gumbel noise for stochastic optimization,
    useful for learning hard permutations.
    """

    def __init__(
        self,
        num_iterations: int = 10,
        temperature: float = 1.0,
        noise_factor: float = 1.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.noise_factor = noise_factor

    def forward(
        self,
        log_scores: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Apply Gumbel-Sinkhorn.

        Args:
            log_scores: Log scores (B, H, T, T)
            add_noise: Whether to add Gumbel noise (set False for inference)

        Returns:
            Doubly stochastic matrix
        """
        if add_noise and self.training:
            # Add Gumbel noise
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(log_scores) + 1e-20) + 1e-20
            )
            log_scores = log_scores + self.noise_factor * gumbel_noise

        # Standard Sinkhorn
        log_scores = log_scores / self.temperature

        for _ in range(self.num_iterations):
            log_scores = log_scores - torch.logsumexp(
                log_scores, dim=-1, keepdim=True
            )
            log_scores = log_scores - torch.logsumexp(
                log_scores, dim=-2, keepdim=True
            )

        return torch.exp(log_scores)
