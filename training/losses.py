"""Loss definitions for end-to-end watermark training."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import DEFAULT_CONFIG
from utils.metrics import structural_similarity


class WatermarkLoss(nn.Module):
    """Combine invisibility and detection losses for watermark training."""

    def __init__(self) -> None:
        super().__init__()
        self.config = DEFAULT_CONFIG.losses
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        original_image: Tensor,
        watermarked_image: Tensor,
        decoded_logits: Tensor,
        target_key: Tensor,
    ) -> dict[str, Tensor]:
        """Return individual loss terms and the weighted total loss."""

        ssim_term = 1.0 - structural_similarity(original_image, watermarked_image)
        l2_term = F.mse_loss(watermarked_image, original_image)
        invisibility_loss = self.config.ssim_weight * ssim_term + self.config.l2_weight * l2_term
        detection_loss = self.bce(decoded_logits, target_key)
        total_loss = (
            self.config.invisibility_weight * invisibility_loss
            + self.config.detection_weight * detection_loss
        )

        return {
            "total_loss": total_loss,
            "invisibility_loss": invisibility_loss,
            "detection_loss": detection_loss,
            "ssim_loss": ssim_term,
            "l2_loss": l2_term,
        }
