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


class TCNmodel(torch.nn.Module):
    def __init__(self, args):
        super(TCNmodel, self).__init__()
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

    def forward(self, x):
        # reshape the x of (batch_size, seq_len, input_size) to (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        for layer in self.tcn_layers:
            x = layer(x)
        x = x[:, :, -1]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNClassifier(torch.nn.Module):
    def __init__(self, args):
        super(CNNClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(args.input_size, args.hidden_size)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=args.input_size, out_channels=args.num_filters, kernel_size=fs)
            for fs in args.filter_sizes
        ])
        self.fc = torch.nn.Linear(len(args.filter_sizes) * args.num_filters, args.output_size)

    def forward(self, x):
        x = x.transpose(1, 2)

        # Apply 1D convolution and ReLU activation for each convolutional layer
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]

        # Apply max pooling to the convolutional layer outputs
        max_pool_outputs = [F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]) for conv_out in conv_outputs]

        # Concatenate max pooling outputs and flatten
        # max_pool_outputs dim: (batch_size, num_filters, 1)
        concatenated_outputs = torch.cat(max_pool_outputs, dim=1).squeeze(2)
        # print(concatenated_outputs.shape)
        # concatenated_outputs dim: (batch_size, num_filters * len(filter_sizes))

        # Pass the concatenated outputs through the fully connected layer
        logits = self.fc(concatenated_outputs)
        return logits


class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, args):
        super(BiLSTMClassifier, self).__init__()
        # self.embedding = torch.nn.Embedding(args.input_size, args.embedding_size)
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.lstm = torch.nn.LSTM(args.input_size, args.hidden_size, args.num_layers, batch_first=True,
                                  bidirectional=True)
        self.fc = torch.nn.Linear(args.hidden_size, args.output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # h0: (num_layers * num_directions, batch_size, hidden_size)
        # c0: (num_layers * num_directions, batch_size, hidden_size)
        # x = torch.tensor(x).to(x.device).long()
        # x=self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, (h, c) = self.lstm(x, (h0, c0))
        # out: (batch_size, seq_len, num_directions * hidden_size)
        # h: (num_layers * num_directions, batch_size, hidden_size)
        # c: (num_layers * num_directions, batch_size, hidden_size)
        # label: (batch_size, output_size)
        # take mean on first dimension of h
        h_ = torch.mean(h, dim=0)
        # h_: (batch_size, hidden_size)
        # linear layer
        label = self.fc(h_)
        # label: (batch_size, output_size)

        return label
