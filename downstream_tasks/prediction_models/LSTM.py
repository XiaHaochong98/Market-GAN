import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            args.input_dim, args.hidden_dim, args.num_layers, batch_first=True, dropout=args.dropout
        )

        # Fully connected layer
        self.fc = nn.Linear(args.hidden_dim, args.output_dim)

    def forward(self, x, t):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        out = out.unsqueeze(1)
        return out
