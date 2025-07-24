import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.seq_len = config.forecast.seq_len
        self.pred_len = config.forecast.pred_len

        self.input_dim = config.model_params.enc_in
        self.output_dim = config.model_params.c_out
        self.feature_type = config.data.features
        self.num_channels = getattr(config.model_params, "num_channels", 32)
        self.kernel_size = getattr(config.model_params, "kernel_size", 3)

        # Override output_dim for MS mode
        if self.feature_type == "MS":
            output_dim = 1

        self.conv1 = nn.Conv1d(self.input_dim, self.num_channels, self.kernel_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(self.num_channels, self.num_channels, self.kernel_size)

        conv_output_size = self.num_channels * (self.seq_len - 2 * (self.kernel_size - 1))
        self.fc = nn.Linear(conv_output_size, output_dim * self.pred_len)

        self.output_dim = output_dim
        self.pred_len = self.pred_len

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, seq_len, input_dim] -> [B, input_dim, seq_len]
        x = x_enc.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x.view(x.shape[0], self.pred_len, self.output_dim)  # [B, pred_len, output_dim]