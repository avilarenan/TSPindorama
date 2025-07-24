import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        input_dim = config.model_params.enc_in
        seq_len = config.forecast.seq_len
        pred_len = config.forecast.pred_len
        output_dim = config.model_params.c_out
        feature_type = config.data.features
        hidden_size = getattr(config.model_params, "hidden_size", 128)
        num_layers = getattr(config.model_params, "num_layers", 2)
        dropout = getattr(config.model_params, "dropout", 0.2)

        # Override output_dim for MS mode
        if feature_type == "MS":
            output_dim = 1

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_dim)
        self.output_dim = output_dim
        self.pred_len = pred_len

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, seq_len, input_dim]
        out, _ = self.lstm(x_enc)  # [B, seq_len, hidden_size]
        # Use the last hidden state for prediction and repeat for pred_len
        last_hidden = out[:, -1, :]  # [B, hidden_size]
        repeated = last_hidden.unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, pred_len, hidden_size]
        output = self.fc(repeated)  # [B, pred_len, output_dim]
        return output