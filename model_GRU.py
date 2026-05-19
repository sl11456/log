import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 添加层归一化
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # 批归一化

        # GRU网络层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，与LSTM相同
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 确保隐藏状态与输入张量在同一设备上
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0)
        # 将GRU的最后一个时间步的输出通过全连接层
        out = self.fc(self.batch_norm(out[:, -1, :]))
        return out.reshape(-1,)
