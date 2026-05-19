import torch
import torch.nn as nn

from model_KAN import KAN


class CustomTransformerEncoder(nn.Module):
    def __init__(self, hidden_space, num_heads, num_layers, dropout_rate):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_space,
                nhead=num_heads,
                dropout=dropout_rate
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x):
        # x: (seq_len, batch_size, input_dim)
        for i in range(self.num_layers):
            x_residual = x  # Save the residual (the input to this layer)
            x = self.layers[i](x)  # Apply the transformer encoder layer
            x = x + x_residual  # Add residual connection
        return x


class TimeSeriesTransformer_ekan_large(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_outputs, hidden_space, dropout_rate=0.1, num_encoders=12):
        super(TimeSeriesTransformer_ekan_large, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.hidden_space = hidden_space
        self.num_encoders = num_encoders

        # Create multiple Custom Transformer Encoders with independent layers
        self.transformer_encoders = nn.ModuleList([
            CustomTransformerEncoder(hidden_space, num_heads, num_layers, dropout_rate)
            for _ in range(num_encoders)
        ])

        # Define KAN layers
        self.e_kan_1 = KAN([input_dim, 10, hidden_space])
        self.e_kan_2 = KAN([hidden_space, 10, num_outputs])
        self.batch_norm = nn.BatchNorm1d(hidden_space)

    def forward(self, x):
        # Permute input to match the shape expected by nn.Transformer
        x0 = x.permute(1, 0, 2)  # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)
        x = self.e_kan_1(x0.reshape(x0.shape[0] * x0.shape[1], -1))  # Transform to hidden_space dimension
        x = x.reshape(x0.shape[0], x0.shape[1], -1)

        # Batch normalization
        x = self.batch_norm(x.permute(1, 2, 0)).permute(2, 0, 1)  # (batch_size, num_features, seq_len) -> (seq_len, batch_size, num_features)

        # Pass data through multiple Transformer Encoders
        for encoder in self.transformer_encoders:
            x = encoder(x)

        # Take the output from the last time step
        x = x[-1, :, :]

        # Final KAN layer
        x = self.e_kan_2(x)
        return x.reshape(-1, )