import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, args):
        super(GRU, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            args.input_dim, args.hidden_dim, args.num_layers, batch_first=True, dropout=args.dropout
        )

        # Fully connected layer
        self.fc = nn.Linear(args.hidden_dim, args.output_dim)

    def forward(self, x, t):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        out.unsqueeze_(1)

        return out
