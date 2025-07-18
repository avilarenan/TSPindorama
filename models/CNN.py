import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.cnn = nn.Sequential(
            nn.Conv1d(self.enc_in, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.pred_len)
        )
        self.out_layer = nn.Conv1d(64, self.enc_in, kernel_size=1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = x_enc.permute(0, 2, 1)  # (B, C, T)
        x = self.cnn(x)
        out = self.out_layer(x)
        return out.permute(0, 2, 1)  # (B, T, C)
