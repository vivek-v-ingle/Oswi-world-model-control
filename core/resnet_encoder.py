"""
ResNet-18 Visual Feature Extractor for OSVI-WM.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetFeats(nn.Module):
    def __init__(self, out_dim=256, output_raw=True, drop_dim=2):
        super().__init__()
        # Initialize ResNet-18 architecture directly (weights loaded from checkpoint)
        resnet = models.resnet18(weights=None)
        self._features = nn.Sequential(*list(resnet.children())[:-drop_dim])
        self._output_raw = output_raw
        self._out_dim = 512

    def forward(self, inputs):
        reshaped = len(inputs.shape) == 5
        x = inputs.reshape((-1, inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]))
        out = self._features(x)
        NH, NW = out.shape[-2:]
        out = out.reshape((inputs.shape[0], inputs.shape[1], self._out_dim, NH, NW)) if reshaped else out
        return out


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetFeats(output_raw=True, drop_dim=2)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad = False
        self.feat_dim = 512

    def forward(self, images, context):
        """
        Args:
            images: Tensor of shape (B, T_agent, 3, H, W)
            context: Tensor of shape (B, T_context, 3, H, W)
        """
        assert len(images.shape) == 5, "expects [B, T, C, H, W] tensor!"
        im_in = torch.cat((context, images), dim=1)
        T_ctxt = context.shape[1]

        enc_features = self.encoder(im_in)
        target_im = im_in[:, T_ctxt + 1:]
        with torch.no_grad():
            target_features = self.target_encoder(target_im)

        return enc_features, target_features
