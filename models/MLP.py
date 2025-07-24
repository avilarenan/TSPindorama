
import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.forecast.seq_len
        self.pred_len = configs.forecast.pred_len
        self.enc_in = configs.model_params.enc_in
        self.feature_type = getattr(configs.data, 'features', 'MS')
        self.hidden_dim = getattr(configs.model_params, 'd_model', 512)

        # Override c_out if features == 'MS'
        if self.feature_type == 'MS':
            self.c_out = 1
        else:
            self.c_out = configs.model_params.c_out

        # Define input/output size for MLP
        input_size = self.seq_len * self.enc_in
        output_size = self.pred_len * self.c_out

        # Build the model
        self.model = nn.Sequential(
            nn.Flatten(),                         # [B, seq_len, enc_in] → [B, seq_len * enc_in]
            nn.Linear(input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_size)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        batch_size = x_enc.shape[0]
        out = self.model(x_enc)
        out = out.view(batch_size, self.pred_len, self.c_out)
        return out
