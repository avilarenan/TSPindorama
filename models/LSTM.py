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

    def forward(self, x):  # x: [B, seq_len, input_size]
        output, (h_n, _) = self.lstm(x)  # h_n: [num_layers, B, hidden_size]
        last_hidden = h_n[-1]            # [B, hidden_size] -> last layer's final hidden state
        out = self.linear(last_hidden)   # [B, pred_len]
        return out.unsqueeze(-1)         # [B, pred_len, 1] for single-variate output