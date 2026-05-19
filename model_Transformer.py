import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_space, num_layers, num_outputs, num_heads=4, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space = hidden_space

        # Transformer 的 Encoder 部分
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_space,  # 输入特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # 将 Encoder 的输出通过一个全连接层转换为所需的输出维度
        self.output_layer = nn.Linear(hidden_space, num_outputs)
        self.transform_layer = nn.Linear(input_dim, hidden_space)
        self.batch_norm = nn.BatchNorm1d(hidden_space)  # 批归一化
    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)

        x = x.permute(1, 0, 2)
        x = self.transform_layer(x)
        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 取最后一个时间步的输出
        x = x[-1, :, :]
        x = self.batch_norm(x)  # 批归一化
        # 全连接层生成最终输出
        x = self.output_layer(x)
        out = x.reshape(-1, )
        contains_nan = torch.isnan(out).any()
        if contains_nan == 1:
            a = 1
        return out



