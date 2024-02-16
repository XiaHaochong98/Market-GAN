import torch


class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv1d(x)
        x = x[:, :, :-self.padding]
        return x


class TCNResidualBlock(torch.nn.Module):
    def __init__(self, in_channels,
                 out_channels, kernel_size, dilation):
        super(TCNResidualBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.causal_conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = torch.nn.ReLU()
        if in_channels != out_channels:
            self.skip = torch.nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = None

    def forward(self, x):
        residual = x
        x = self.relu(self.causal_conv1(x))
        x = self.relu(self.causal_conv2(x))

        if self.skip is not None:
            residual = self.skip(residual)

        return self.relu(x + residual)


class TCN(torch.nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        input_size = args.input_size
        output_size = args.output_size
        num_channels = args.num_channels
        kernel_size = args.kernel_size
        dropout = args.dropout
        self.tcn_layers = torch.nn.ModuleList()
        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.tcn_layers.append(
                TCNResidualBlock(in_channels, out_channels, kernel_size, dilation=1)
            )
            self.fc = torch.nn.Linear(num_channels[-1], output_size)
            self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, t):
        # reshape the x of (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        for layer in self.tcn_layers:
            x = layer(x)
        x = x[:, :, -1]
        x = self.dropout(x)
        x = self.fc(x)
        # reshape the x of (batch_size, input_size) to (batch_size, 1, seq_len)
        x = x.unsqueeze(1)
        return x
