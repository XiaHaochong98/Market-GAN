import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv1d(x)
        x = x[:, :, :-self.padding]
        return x


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels, kernel_size, dilation):
        super(TCNResidualBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.causal_conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = None

    def forward(self, x):
        residual = x
        x = self.relu(self.causal_conv1(x))
        x = self.relu(self.causal_conv2(x))

        if self.skip is not None:
            residual = self.skip(residual)

        return self.relu(x + residual)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn_layers = nn.ModuleList()
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.tcn_layers.append(
                TCNResidualBlock(in_channels, out_channels, kernel_size, dilation=2 ** i)
            )
            self.fc = nn.Linear(num_channels[-1], output_size)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # transpose x to (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        for layer in self.tcn_layers:
            x = layer(x)
        x = x[:, :, -1]
        x = self.dropout(x)
        x = self.fc(x)
        # softmax x
        x = torch.softmax(x, dim=1)
        return x
