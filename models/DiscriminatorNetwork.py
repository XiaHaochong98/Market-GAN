# -*- coding: UTF-8 -*-
import torch

from models.TimesNet import TimesNet


class DiscriminatorNetwork(torch.nn.Module):
    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.use_TimesNet = args.use_TimesNet
        self.use_RNN = args.use_RNN
        if self.use_TimesNet:
            args.rnn_input_dim = args.hidden_dim
            args.enc_in = args.hidden_dim
        else:
            args.rnn_input_dim = args.hidden_dim
        self.args = args
        if self.args.add_history > 0:
            self.args.condition_dim += self.args.hidden_size
        self.model = DiscriminatorRNN(args)
        if self.use_TimesNet:
            self.timesnet = TimesNet(self.args)
        print(
            f'Discriminator use TimesNet: {self.use_TimesNet}, addtional RNN: {self.use_RNN}, add_history:{self.args.add_history}')

    def forward(self, X, T, D=None, L=None, H=None):
        if self.use_TimesNet:
            # broadcast D and L to the same shape as X
            D_ = D.unsqueeze(1).repeat(1, X.shape[1], 1)
            L_ = L.unsqueeze(1).repeat(1, X.shape[1], 1)
            # concatenate D and L to C
            C = torch.cat((D_, L_), dim=2)
        if H is not None:
            if self.args.add_history == 2:
                H_ = H
                if self.use_TimesNet:
                    C = torch.cat((C, H_), dim=2)
            elif self.args.add_history == 1:
                H_ = H
            else:
                H_ = None
        else:
            H_ = None
        if self.use_TimesNet:
            X = self.timesnet(X, C)
        if self.use_RNN:
            X = self.model.forward(X, T, D, L, H_)
        return X


class DiscriminatorRNN(torch.nn.Module):
    """The time feature Discriminator network (decoder) for TimeGAN
    """

    def __init__(self, args):
        super(DiscriminatorRNN, self).__init__()
        self.rnn_input_dim = args.rnn_input_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        if args.add_history > 0:
            history_embedding_dim = args.hidden_size
        else:
            history_embedding_dim = 0
        if args.conditional == True:
            self.dis_conditional_linear = torch.nn.Linear(args.dynamic_dim + args.label_dim, self.hidden_dim)
            self.dis_linear = torch.nn.Linear(self.hidden_dim * 2 + history_embedding_dim, 1)
        else:
            self.dis_linear = torch.nn.Linear(self.hidden_dim + history_embedding_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T, D, L, History=None):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # Dynamic RNN input for ignoring paddings
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.dis_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        if D is not None or L is not None:
            C = torch.cat((D, L), 1)
            conditional_embedding = self.dis_conditional_linear(C)
            H_o = torch.cat((H_o, conditional_embedding.unsqueeze(1).repeat(1, H_o.shape[1], 1)), 2)
        if History is not None:
            H_o = torch.cat((H_o, History), 2)

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits
