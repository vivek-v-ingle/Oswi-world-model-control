"""
Spatio-Temporal Non-Local Attention Layers for Action/Intent Modeling.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer("pe", pe)

    def forward(self, x):
        assert len(x.shape) >= 3, "x requires at least 3 dims! (B, C, ..)"
        old_shape = x.shape
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x + self.pe[:, :x.shape[1], :x.shape[-1]]
        return self.dropout(x).reshape(old_shape)


class NonLocalLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        feedforward_dim=512,
        dropout=0,
        temperature=None,
        causal=False,
        n_heads=1,
        dino_feat=False,
    ):
        super().__init__()
        assert feedforward_dim % n_heads == 0, "n_heads must evenly divide feedforward_dim"
        self._n_heads = n_heads
        self._temperature = temperature if temperature is not None else np.sqrt(in_dim)
        if dino_feat:
            self._K = nn.Sequential(
                nn.Conv3d(in_dim, feedforward_dim, 1, stride=1, bias=False),
                nn.MaxPool3d(kernel_size=(1, 10, 10), stride=(1, 1, 1)),
            )
            self._V = nn.Sequential(
                nn.Conv3d(in_dim, feedforward_dim, 1, stride=1, bias=False),
                nn.MaxPool3d(kernel_size=(1, 10, 10), stride=(1, 1, 1)),
            )
            self._Q = nn.Sequential(
                nn.Conv3d(in_dim, feedforward_dim, 1, stride=1, bias=False),
                nn.MaxPool3d(kernel_size=(1, 10, 10), stride=(1, 1, 1)),
            )
            self._skip = False
        else:
            self._K = nn.Conv3d(in_dim, feedforward_dim, 1, stride=1, bias=False)
            self._V = nn.Conv3d(in_dim, feedforward_dim, 1, stride=1, bias=False)
            self._Q = nn.Conv3d(in_dim, feedforward_dim, 1, stride=1, bias=False)
            self._skip = out_dim == in_dim

        self._out = nn.Conv3d(feedforward_dim, out_dim, 1)
        self._a1 = nn.ReLU(inplace=dropout == 0)
        self._drop1 = nn.Dropout3d(dropout)
        self._norm = nn.BatchNorm3d(out_dim)
        self._causal = causal

    def forward(self, inputs):
        K, Q, V = self._K(inputs), self._Q(inputs), self._V(inputs)
        B, C, T, H, W = K.shape
        K, Q, V = [
            t.reshape((B, self._n_heads, int(C / self._n_heads), T * H * W))
            for t in (K, Q, V)
        ]
        KQ = torch.matmul(K.transpose(2, 3), Q) / self._temperature
        if self._causal:
            mask = torch.tril(torch.ones((T, T))).to(KQ.device)
            mask = mask.repeat_interleave(H * W, 0).repeat_interleave(H * W, 1)
            KQ = KQ + torch.log(mask + 1e-9).unsqueeze(0).unsqueeze(0)
        attn = F.softmax(KQ, 3)
        V = torch.matmul(V, attn.transpose(2, 3)).reshape((B, C, T, H, W))
        out = (
            inputs + self._drop1(self._a1(self._out(V)))
            if self._skip
            else self._drop1(self._a1(self._out(V)))
        )
        return self._norm(out)
