import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.hidden_size = configs.d_model

        self.lstm = nn.LSTM(input_size=self.enc_in,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True)
        
        self.linear = nn.Linear(self.hidden_size, self.enc_in)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        output, _ = self.lstm(x_enc)
        last_hidden = output[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        out = self.linear(last_hidden)
        return out
