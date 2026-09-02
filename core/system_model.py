"""
Transformer-based Forward Dynamics World Model.
Autoregressively predicts next latent states given current state and action representations.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from core.utils.modules import Block


class LayerNorm(nn.Module):
    """LayerNorm with optional bias support."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        window_size=30,
        n_layer=1,
        n_head=4,
        n_embed=512,
        output_dim=None,
        dropout=0.3,
        bias=True,
        is_causal=True,
    ):
        super().__init__()
        assert input_dim is not None
        assert window_size is not None
        if output_dim is None:
            output_dim = input_dim // 2

        self.window_size = window_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(input_dim, n_embed),
                wpe=nn.Embedding(window_size, n_embed),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([
                    Block(dim=n_embed, num_heads=n_head, drop=0.0, attn_drop=0.0, is_causal=is_causal)
                    for _ in range(n_layer)
                ]),
                ln_f=LayerNorm(n_embed, bias=bias),
            )
        )
        self.output_head = nn.Linear(n_embed, output_dim, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target=None):
        device = x.device
        b, t, d = x.size()
        assert t <= self.window_size, f"Sequence length {t} exceeds window size {self.window_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        output = self.output_head(x)

        if target is None:
            return output
        loss = F.mse_loss(output, target)
        return output, loss
