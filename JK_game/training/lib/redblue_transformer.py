"The transformer used for RedBule"

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.lib.sublayers import MultiHeadAttention, PositionwiseFeedForward


class Transformer(nn.Module):
    def __init__(
            self, d_model=128, d_inner=512,
            n_layers=3, n_head=2, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        enc_output, *_ = self.encoder(x)

        return enc_output


class Encoder(nn.Module):
    ''' A alphastar encoder model with self attention mechanism '''

    def __init__(
            self, n_layers=3, n_head=2, d_k=64, d_v=64,
            d_model=128, d_inner=512, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, d_inner=512, n_head=2, d_k=64, d_v=64, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
