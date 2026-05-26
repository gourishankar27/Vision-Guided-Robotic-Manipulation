from __future__ import annotations

import torch
from torch import Tensor, nn

from src.vision_encoder import VisionEncoder


class TemporalVisionFrankaPolicy(nn.Module):
    """Temporal RGB+state policy for Isaac Franka sequence prediction.

    Inputs are K context frames/states. Outputs are H future joint deltas and
    H future end-effector positions. This is a stronger bridge than a single
    transition model because it can infer phase of the pick/place behavior and
    optimize multi-step losses.
    """

    def __init__(
        self,
        dof: int,
        context_len: int = 4,
        horizon: int = 8,
        image_channels: int = 3,
        feature_dim: int = 192,
        base_channels: int = 24,
        state_dim: int | None = None,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.05,
        max_delta: float = 0.25,
        vision_dropout: float = 0.0,
        state_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dof = int(dof)
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.max_delta = float(max_delta)
        self.vision_dropout = float(vision_dropout)
        self.state_dropout = float(state_dropout)
        self.encoder = VisionEncoder(in_channels=image_channels, feature_dim=feature_dim, base_channels=base_channels)
        # q, current EE, cube, target, and scalar time index in context.
        state_dim = int(state_dim or (self.dof + 3 + 3 + 3 + 1))
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.vision_proj = nn.Sequential(nn.Linear(feature_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(inplace=True))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.context_len, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.summary = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.SiLU(inplace=True))
        self.delta_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, self.horizon * self.dof),
        )
        self.ee_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, self.horizon * 3),
        )
        self.value_head = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.SiLU(inplace=True), nn.Linear(embed_dim // 2, 1))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, frames: Tensor, q_context: Tensor, ee_context: Tensor, cube_context: Tensor, target: Tensor) -> dict[str, Tensor]:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B, K, C, H, W]")
        B, K, C, H, W = frames.shape
        if K != self.context_len:
            raise ValueError(f"Expected context_len={self.context_len}, got {K}")
        if q_context.shape[:2] != (B, K):
            raise ValueError("q_context must have shape [B, K, dof]")
        vf = self.encoder(frames.reshape(B * K, C, H, W)).reshape(B, K, -1)
        if self.training and self.vision_dropout > 0.0:
            keep = (torch.rand(B, K, 1, device=frames.device) >= self.vision_dropout).to(vf.dtype)
            vf = vf * keep
        target_rep = target[:, None, :].expand(B, K, 3)
        # Normalized time-in-context token helps the transformer know order even before pos_embed.
        t = torch.linspace(0.0, 1.0, K, device=frames.device, dtype=frames.dtype).view(1, K, 1).expand(B, K, 1)
        state = torch.cat([q_context, ee_context, cube_context, target_rep, t], dim=-1)
        if self.training and self.state_dropout > 0.0:
            keep_state = (torch.rand(B, K, 1, device=frames.device) >= self.state_dropout).to(state.dtype)
            state = state * keep_state
        tokens = self.vision_proj(vf) + self.state_proj(state) + self.pos_embed[:, :K]
        tokens = self.temporal(tokens)
        h = self.summary(tokens[:, -1])
        delta = self.max_delta * torch.tanh(self.delta_head(h)).reshape(B, self.horizon, self.dof)
        ee = self.ee_head(h).reshape(B, self.horizon, 3)
        value = self.value_head(h).squeeze(-1)
        return {"delta_q_seq": delta, "ee_seq": ee, "value": value}
