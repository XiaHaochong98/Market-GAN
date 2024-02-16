# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

from models.TimesNet import TimesNet


class GeneratorNetwork(torch.nn.Module):
    def __init__(self, args):
        super(GeneratorNetwork, self).__init__()
        self.use_TimesNet = args.use_TimesNet
        self.use_RNN = args.use_RNN
        if self.use_TimesNet:
            args.rnn_feature_dim = args.hidden_dim
            args.enc_in = args.Z_dim
        else:
            args.rnn_feature_dim = args.Z_dim
        self.model = GeneratorRNN(args)
        self.activation_fn = nn.Sigmoid()
        self.args = args

        if self.args.add_history > 0:
            # self.args.condition_dim+=self.HistoryEmbeddingNetwork.history_embedding_dim
            self.args.condition_dim += self.args.hidden_size
        if self.use_TimesNet:
            self.timesnet = TimesNet(self.args)
        print(
            f'Generator use TimesNet: {self.use_TimesNet}, addtional RNN: {self.use_RNN}, add_history:{self.args.add_history}')

    def forward(self, Z, T, D=None, L=None, H=None):
        if self.use_TimesNet:
            # broadcast D and L to the same shape as X
            D_ = D.unsqueeze(1).repeat(1, Z.shape[1], 1)
            L_ = L.unsqueeze(1).repeat(1, Z.shape[1], 1)
            # concatenate D and L to C
            C = torch.cat((D_, L_), dim=2)
        if H is not None:
            if self.args.add_history == 2:
                # H_ = self.HistoryEmbeddingNetwork(H)
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
            Z = self.timesnet(Z, C)
        if self.use_RNN:
            Z = self.model.forward(Z, T, D, L, H_)
        Z = self.activation_fn(Z)
        return Z


class GeneratorRNN(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """

    def __init__(self, args):
        super(GeneratorRNN, self).__init__()
        self.Z_dim = args.rnn_feature_dim
        self.hidden_dim = args.hidden_dim
        # if args.conditional==True:
        #     self.label_embedding = torch.nn.Embedding(args.dynamic_dim+args.label_dim, args.embedding_dim)
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.dynamic_dim = args.dynamic_dim
        self.label_dim = args.label_dim
        self.embedding_dim = args.embedding_dim

        self.gen_rnn = torch.nn.GRU(
            input_size=self.Z_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        if args.add_history > 0:
            history_embedding_dim = args.hidden_size
        else:
            history_embedding_dim = 0
        if args.conditional == True:
            self.gen_conditional_linear = torch.nn.Linear(args.dynamic_dim + args.label_dim, self.hidden_dim)
            self.gen_linear = torch.nn.Linear(self.hidden_dim * 2 + history_embedding_dim, self.hidden_dim)
        else:
            self.gen_linear = torch.nn.Linear(self.hidden_dim + history_embedding_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, Z, T, D=None, L=None, History=None):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
            - D: input dynamic features (B x D)
            - L: input label features (B x L)
        Returns:
            - H: embeddings (B x S x E)
        """
        # concate D and L and embed them
        # if D is not None or L is not None:
        #     label=torch.cat((D,L),1)
        #     #broadcast label to the same size as Z
        #     label=label.unsqueeze(1).repeat(1,self.max_seq_len,1)
        #     #concatenate Z and label
        #     # print(Z.shape,label.shape)
        #     Z=torch.cat((Z,label),2)

        # Dynamic RNN input for ignoring paddings
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 71
        H_o, H_t = self.gen_rnn(Z_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        if D is not None or L is not None:
            C = torch.cat((D, L), 1)
            conditional_embedding = self.gen_conditional_linear(C)
            H_o = torch.cat((H_o, conditional_embedding.unsqueeze(1).repeat(1, H_o.shape[1], 1)), 2)
        if History is not None:
            H_o = torch.cat((H_o, History), 2)
        # 128 x 100 x 10
        H = self.gen_linear(H_o)
        # B x S
        return H
