import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        input_dim = config.model_params.enc_in
        seq_len = config.forecast.seq_len
        pred_len = config.forecast.pred_len
        output_dim = config.model_params.c_out
        feature_type = config.data.features

        # Override output_dim for MS mode
        if feature_type == "MS":
            output_dim = 1

        d_model = getattr(config.model_params, "d_model", 512)
        n_heads = getattr(config.model_params, "n_heads", 8)
        num_layers = getattr(config.model_params, "e_layers", 3)
        dim_feedforward = getattr(config.model_params, "d_ff", 2048)
        dropout = getattr(config.model_params, "dropout", 0.1)

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model * seq_len, pred_len * output_dim)

        self.pred_len = pred_len
        self.output_dim = output_dim

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, seq_len, input_dim]
        x = self.input_projection(x_enc)  # [B, seq_len, d_model]
        x = self.encoder(x)               # [B, seq_len, d_model]
        x = x.flatten(start_dim=1)       # [B, seq_len * d_model]
        x = self.output_layer(x)         # [B, pred_len * output_dim]
        return x.view(x.shape[0], self.pred_len, self.output_dim)  # [B, pred_len, output_dim]
