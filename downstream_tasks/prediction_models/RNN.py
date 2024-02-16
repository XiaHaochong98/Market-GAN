import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        # RNN layers
        self.rnn = nn.RNN(
            args.input_dim, args.hidden_dim, args.num_layers, batch_first=True, dropout=args.dropout
        )
        # Fully connected layer
        self.fc = nn.Linear(args.hidden_dim, args.output_dim)

    def forward(self, x, t):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        out = self.fc(out)
        # transform to the output to shape ( batch_size, 1, output_dim)
        out = out.unsqueeze(1)
        return out
