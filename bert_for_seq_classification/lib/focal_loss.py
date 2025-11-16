"""Focal Loss implementation for PyTorch.

This implements the multiclass focal loss as described in "Focal Loss for Dense
Object Detection" (Lin et al.), adapted for classification tasks https://arxiv.org/abs/1708.02002.

Usage:
    from focal_loss import FocalLoss
    loss_fn = FocalLoss(gamma=2.0, alpha=None, reduction='mean')
    loss = loss_fn(logits, labels)

Parameters:
    gamma: focusing parameter (>=0). Higher gamma down-weights easy examples.
    alpha: class weighting. Can be None, a float, or a sequence of length num_classes.
    reduction: one of 'none', 'mean', 'sum'.
    ignore_index: label value to ignore when computing loss (like in CrossEntropyLoss).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass Focal Loss.

    Args:
        gamma: focusing parameter.
        alpha: class weighting. If float, multiplies the loss for all classes by that factor.
               If sequence/tensor, should have shape (num_classes,) and assign per-class weights.
        reduction: 'none' | 'mean' | 'sum'
        ignore_index: target value to ignore.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is None:
            self.register_buffer("alpha", torch.tensor(1.0))
        else:
            if isinstance(alpha, float) or isinstance(alpha, int):
                self.register_buffer("alpha", torch.tensor(float(alpha)))
            else:
                a = torch.tensor(list(alpha), dtype=torch.float)
                self.register_buffer("alpha", a)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Tensor of shape (N, C).
            targets: LongTensor of shape (N,) with class indices.
        """
        if logits.dim() != 2:
            raise ValueError("FocalLoss expects logits with shape (N, C)")

        targets = targets.long()

        log_probs = F.log_softmax(logits, dim=1)  # (N, C)
        probs = torch.exp(log_probs)

        # Gather log-probabilities and probabilities of the target class
        targets_unsq = targets.unsqueeze(1)
        log_pt = log_probs.gather(1, targets_unsq).squeeze(1)  # (N,)
        pt = probs.gather(1, targets_unsq).squeeze(1)  # (N,)

        # Focal factor
        focal_factor = (1.0 - pt) ** self.gamma

        loss = -focal_factor * log_pt

        # Apply alpha (class weight). If alpha is a scalar, multiply all examples.
        if self.alpha is not None:
            if self.alpha.ndim == 0:
                loss = loss * self.alpha
            else:
                # per-sample factor from class weight
                loss = loss * self.alpha[targets]

        # Handle ignore_index
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            loss = loss * valid_mask.to(loss.dtype)
            denom = valid_mask.sum()
        else:
            denom = torch.tensor(targets.numel(), dtype=loss.dtype, device=loss.device)

        if self.reduction == "mean":
            denom_val = denom if isinstance(denom, torch.Tensor) else torch.tensor(denom, device=loss.device)
            if denom_val.item() == 0:
                return loss.sum() * 0.0
            return loss.sum() / denom_val
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


__all__ = ["FocalLoss"]
