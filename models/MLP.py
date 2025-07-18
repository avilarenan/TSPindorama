import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.enc_in, 512),
            nn.ReLU(),
            nn.Linear(512, self.pred_len * self.enc_in)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = self.flatten(x_enc)
        out = self.fc(x)
        return out.view(out.shape[0], self.pred_len, self.enc_in)
