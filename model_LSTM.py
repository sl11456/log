import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)  # 添加层归一化
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # 批归一化
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.layer_norm(out[:, -1, :])  # 对 LSTM 的输出进行层归一化
        out = self.batch_norm(out[:, -1, :])  # 批归一化
        out = self.fc(out)
        out = out.reshape(-1,)
        return out