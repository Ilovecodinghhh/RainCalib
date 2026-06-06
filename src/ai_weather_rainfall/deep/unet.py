from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        channels = [base_channels * (2**level) for level in range(depth + 1)]
        self.down_blocks = nn.ModuleList()
        current = in_channels
        for channel_count in channels:
            self.down_blocks.append(ConvBlock(current, channel_count, dropout=dropout))
            current = channel_count
        self.pool = nn.MaxPool2d(2)
        self.up_blocks = nn.ModuleList()
        for level in range(depth - 1, -1, -1):
            self.up_blocks.append(
                ConvBlock(channels[level + 1] + channels[level], channels[level], dropout=dropout)
            )
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        for block in self.down_blocks[:-1]:
            out = block(out)
            skips.append(out)
            out = self.pool(out)
        out = self.down_blocks[-1](out)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = block(out)
        return self.output(out)
