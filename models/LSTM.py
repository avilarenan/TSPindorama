import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        input_dim = config.model_params.enc_in
        self.seq_len = config.forecast.seq_len
        self.pred_len = config.forecast.pred_len
        self.output_dim = config.model_params.c_out
        self.feature_type = config.data.features
        self.hidden_size = getattr(config.model_params, "d_model", 128)
        self.num_layers = getattr(config.model_params, "e_layers", 2)
        self.dropout = getattr(config.model_params, "dropout", 0.2)

        # Override output_dim for MS mode
        if self.feature_type == "MS":
            self.output_dim = 1

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.output_dim * self.pred_len)
        self.output_dim = self.output_dim
        
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):  # x: [B, seq_len, input_size]
        out, (h_n, _) = self.lstm(batch_x)         # h_n: [num_layers, B, hidden_size]
        last_hidden = h_n[-1]                # [B, hidden_size]
        out = self.linear(last_hidden)   # [B, output_dim * pred_len]
        return out.view(batch_x.shape[0], self.pred_len, self.output_dim)  # [B, pred_len, output_dim]