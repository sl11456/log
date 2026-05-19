import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Layer normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim*2)  # Batch normalization

        # Bidirectional LSTM network layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # Note: Because it's bidirectional, the input to the fully connected layer is hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device).requires_grad_()
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device).requires_grad_()
        # Forward propagate the bidirectional LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Take the output of the last time step for both directions
        out = self.batch_norm(out[:, -1, :])  # Batch normalization
        out = self.fc(out)
        return out.reshape(-1,)


