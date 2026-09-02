"""
Top-level OSVI-WM Architecture (One-Shot Visual Imitation with World Model).
Combines:
  1. Visual Encoder (ResNet-18)
  2. Spatio-Temporal Action/Intent Model (NonLocalLayer)
  3. Autoregressive Transformer World Model
  4. Spatial Keypoint Soft-Argmax Pooling
  5. Attentive Query Trajectory Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from core.resnet_encoder import VisualEncoder
from core.traj_embed import NonLocalLayer, TemporalPositionalEncoding
from core.system_model import TransformerModel
from core.attentive_pooler import AttentivePooler


class ActionModel(nn.Module):
    def __init__(self, dropout=0.3, attn_heads=4, n_st_attn=6):
        super().__init__()
        self.feat_dim = 512
        self._pe = TemporalPositionalEncoding(self.feat_dim, dropout)
        self._st_attn = nn.Sequential(
            NonLocalLayer(self.feat_dim, 512, 128, dropout=dropout, causal=True, n_heads=attn_heads),
            *[
                NonLocalLayer(512, 512, 128, dropout=dropout, causal=True, n_heads=attn_heads)
                for _ in range(n_st_attn - 2)
            ],
            NonLocalLayer(512, self.feat_dim, 128, dropout=dropout, causal=True, n_heads=attn_heads),
        )

    def forward(self, enc_features):
        enc_features2 = self._pe(enc_features.transpose(1, 2)).transpose(1, 2)
        act_features = self._st_attn(enc_features2.transpose(1, 2)).transpose(1, 2)
        return act_features


class OSVIWorldModel(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        waypoints=5,
        sub_waypoints=True,
        image_resolution=(224, 224),
    ):
        super().__init__()
        if sub_waypoints and waypoints is not None:
            self.total_waypoints = (waypoints + 1) * waypoints // 2
        else:
            self.total_waypoints = waypoints

        self.latent_dim = latent_dim
        self._embed = VisualEncoder()
        self.action_model = ActionModel(dropout=0.3, attn_heads=4, n_st_attn=6)

        in_dim = 512
        # Feature map factor: (224/32)^2 = 7*7 = 49
        H_feat = image_resolution[0] // 32
        W_feat = image_resolution[1] // 32
        self.factor = H_feat * W_feat

        self.forward_model = TransformerModel(
            input_dim=in_dim * self.factor * 2,
            window_size=30,
            n_layer=1,
            n_head=4,
            n_embed=512,
            output_dim=in_dim * self.factor,
            dropout=0.3,
            bias=True,
            is_causal=True,
        )

        self._to_embed = nn.Sequential(nn.Linear(in_dim * 2, latent_dim * 5))
        self.attn_pe = TemporalPositionalEncoding(d_model=in_dim * 2)
        self.attn_pool = AttentivePooler(
            num_queries=1, embed_dim=in_dim * 2, num_heads=1, complete_block=False
        )

        self.waypoint_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(latent_dim * 5, self.total_waypoints * 4),
            Rearrange("... (w d) -> ... w d", d=4),
        )

    def _spatial_embed(self, features):
        """Soft-argmax pooling to extract 2D keypoint coordinates."""
        b, t, c, h, w = features.shape
        flat = features.reshape(b, t, c, -1)
        soft = F.softmax(flat, dim=3).reshape(features.shape)

        h_coord = torch.sum(
            torch.linspace(-1, 1, h, device=features.device).view(1, 1, 1, -1)
            * torch.sum(soft, dim=4),
            dim=3,
        )
        w_coord = torch.sum(
            torch.linspace(-1, 1, w, device=features.device).view(1, 1, 1, -1)
            * torch.sum(soft, dim=3),
            dim=3,
        )
        return torch.cat((h_coord, w_coord), dim=2)

    def forward(self, images, context, T_tot=16):
        """
        Args:
            images: Current agent observation tensor (B, T_obs, 3, H, W).
            context: Teacher video demonstration context (B, T_context, 3, H, W).
            T_tot: Total rollout horizon steps.
        Returns:
            Dictionary containing:
                - 'waypoints': Tensor (B, num_waypoints, 4) -> (u, v, depth, gripper)
                - 'forward_states': Latent rolled out states
        """
        T_context = context.shape[1]
        enc_features, target_features = self._embed(images, context)

        context_states = enc_features[:, :T_context]
        current_state = enc_features[:, T_context : T_context + 1]
        all_next_states = []

        # Autoregressive forward dynamics rollout
        for _ in range(T_tot - T_context - 1):
            act_input = torch.cat([context_states, current_state], dim=1)
            act_features = self.action_model(act_input)

            forward_input = torch.cat(
                [act_features[:, T_context:], current_state], dim=2
            ).flatten(2, 4)

            next_state = self.forward_model(forward_input)[:, -1:]
            next_state = next_state.view(
                next_state.shape[0],
                1,
                enc_features.shape[2],
                enc_features.shape[3],
                enc_features.shape[4],
            )
            current_state = torch.cat([current_state, next_state], dim=1)
            all_next_states.append(next_state)

        all_next_states = torch.cat(all_next_states, dim=1)
        forward_states = self._spatial_embed(all_next_states)
        waypoint_states = forward_states

        # Attentive query pooling to decode waypoints
        waypoint_states = self.attn_pe(waypoint_states)
        waypoint_states = self.attn_pool(waypoint_states)
        waypoint_states = waypoint_states.view(waypoint_states.shape[0], -1)

        act_embed = self._to_embed(waypoint_states)
        waypoints = self.waypoint_head(act_embed)

        return {
            "waypoints": waypoints,
            "forward_states": forward_states,
            "raw_forward_states": all_next_states,
            "enc_features": enc_features,
        }
