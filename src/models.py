from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshPoseScoringModel(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 9,
        base_width: int = 32,
        hidden_dim: int = 256,
        predict_refinement: bool = True,
        translation_refinement_scale: float = 0.0,
        rotation_only_refinement: bool = True,
    ):
        super().__init__()
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        blocks = []
        in_channels = input_channels
        for block_index, out_channels in enumerate(widths):
            stride = 2 if block_index > 0 else 1
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.predict_refinement = predict_refinement
        self.refinement_head = nn.Linear(hidden_dim, 7) if predict_refinement else None
        self.translation_refinement_scale = translation_refinement_scale
        self.rotation_only_refinement = rotation_only_refinement

    def forward(self, pair_tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        features = self.encoder(pair_tensor)
        pooled = self.pool(features)
        embedding = self.projection(pooled)
        score_logit = self.score_head(embedding).squeeze(-1)
        if not self.predict_refinement:
            return score_logit, None
        refinement = self.refinement_head(embedding)
        delta_quaternion = F.normalize(refinement[:, :4], dim=-1)
        if self.rotation_only_refinement:
            delta_translation = torch.zeros_like(refinement[:, 4:])
        else:
            delta_translation = self.translation_refinement_scale * torch.tanh(refinement[:, 4:])
        return score_logit, (delta_quaternion, delta_translation)
