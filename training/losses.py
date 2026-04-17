"""Loss definitions for end-to-end watermark training."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import DEFAULT_CONFIG
from utils.metrics import structural_similarity


class WatermarkLoss(nn.Module):
    """Combine invisibility and detection losses for watermark training."""

    def __init__(
        self,
        invisibility_weight: float = DEFAULT_CONFIG.losses.invisibility_weight,
        detection_weight: float = DEFAULT_CONFIG.losses.detection_weight,
        ssim_weight: float = DEFAULT_CONFIG.losses.ssim_weight,
        l2_weight: float = DEFAULT_CONFIG.losses.l2_weight,
    ) -> None:
        super().__init__()
        self.invisibility_weight = invisibility_weight
        self.detection_weight = detection_weight
        self.ssim_weight = ssim_weight
        self.l2_weight = l2_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        original_image: Tensor,
        watermarked_image: Tensor,
        decoded_logits: Tensor,
        target_key: Tensor,
    ) -> dict[str, Tensor]:
        """Return individual loss terms and the weighted total loss."""

        ssim_value = structural_similarity(original_image, watermarked_image).clamp(0.0, 1.0)
        ssim_term = 1.0 - ssim_value
        l2_term = F.mse_loss(watermarked_image, original_image)
        invisibility_loss = self.ssim_weight * ssim_term + self.l2_weight * l2_term
        detection_loss = self.bce(decoded_logits, target_key)
        total_loss = (
            self.invisibility_weight * invisibility_loss
            + self.detection_weight * detection_loss
        )

        return {
            "total_loss": total_loss,
            "invisibility_loss": invisibility_loss,
            "detection_loss": detection_loss,
            "ssim_loss": ssim_term,
            "l2_loss": l2_term,
        }
