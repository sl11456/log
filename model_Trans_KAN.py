import torch.nn as nn
from model_KAN import KAN


class TimeSeriesTransformer_ekan(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate=0.1):
        super(TimeSeriesTransformer_ekan, self).__init__()
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
        self.e_kan = KAN([hidden_space, 10, num_outputs])
        self.transform_layer = nn.Linear(input_dim, hidden_space)

        # 添加批归一化层
        self.batch_norm = nn.BatchNorm1d(hidden_space)

    def forward(self, x):
        # 转换输入数据维度以符合 Transformer 的要求：(seq_len, batch_size, feature_dim)
        x = x.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, input_dim)
        x = self.transform_layer(x)  # 线性变换到 hidden_space 维度

        # 批归一化
        x = self.batch_norm(x.permute(1, 2, 0)).permute(2, 0, 1)  # 批归一化需要 (batch_size, num_features, seq_len) 形式

        # Transformer 编码器
        x = self.transformer_encoder(x)

        # 取最后一个时间步的输出
        x = x[-1, :, :]

        # 全连接层生成最终输出
        x = self.e_kan(x)
        return x.reshape(-1, )
