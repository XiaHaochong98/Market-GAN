# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

from models.TimesNet import TimesNet


class EmbeddingNetwork(torch.nn.Module):
    def __init__(self, args):
        super(EmbeddingNetwork, self).__init__()
        self.use_TimesNet = args.use_TimesNet
        self.use_RNN = args.use_RNN
        if self.use_TimesNet:
            args.rnn_feature_dim = args.hidden_dim
        else:
            args.rnn_feature_dim = args.feature_dim
        self.activation_fn = nn.Sigmoid()
        self.model = EmbeddingRNN(args)
        self.args = args
        if self.args.add_history > 0:
            self.args.condition_dim += self.args.hidden_size
        if self.use_TimesNet:
            self.timesnet = TimesNet(self.args)
        print(
            f'EmbeddingNetwork use TimesNet: {self.use_TimesNet}, addtional RNN: {self.use_RNN}, add_history:{self.args.add_history}')

    def forward(self, X, T, D=None, L=None, H=None):
        if self.use_TimesNet:
            # broadcast D and L to the same shape as X
            D_ = D.unsqueeze(1).repeat(1, X.shape[1], 1)
            L_ = L.unsqueeze(1).repeat(1, X.shape[1], 1)
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
            X = self.timesnet(X, C)
        if self.use_RNN:
            X = self.model.forward(X, T, D, L, H_)
        X = self.activation_fn(X)
        return X


class EmbeddingRNN(torch.nn.Module):
    """The embedding network (encoder) for TimeGAN
    """

    def __init__(self, args):
        super(EmbeddingRNN, self).__init__()
        self.feature_dim = args.rnn_feature_dim

        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.emb_rnn = torch.nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        if args.add_history > 0:
            history_embedding_dim = args.hidden_size
        else:
            history_embedding_dim = 0
        if args.conditional == True:
            self.emb_conditional_linear = torch.nn.Linear(args.dynamic_dim + args.label_dim, self.hidden_dim)
            self.emb_linear = torch.nn.Linear(self.hidden_dim * 2 + history_embedding_dim, self.hidden_dim)
        else:
            self.emb_linear = torch.nn.Linear(self.hidden_dim + history_embedding_dim, self.hidden_dim)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.emb_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, X, T, D=None, L=None, History=None):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F)
            - T: input temporal information (B)
            - D: input dynamic information (B x D)
            - L: input label information (B x L)
        Returns:
            - H: latent space embeddings (B x S x H)
        """
        # concate D and L and embed them
        # if D is not None or L is not None:
        #     # print("D",D.shape,"L",L.shape)
        #     label=torch.cat((D,L),1)
        #     #broadcast label to the same size as X
        #     label=label.unsqueeze(1).repeat(1,X.shape[1],1)
        #     #concatenate label and X
        #     X=torch.cat((X,label),2)
        # print(X.shape)

        # Dynamic RNN input for ignoring paddings
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 71
        H_o, H_t = self.emb_rnn(X_packed)
        # dim of H_o: 128 x 100 x 10 (batch_size x seq_len x hidden_dim)
        # dim of H_t: 1 x 128 x 10 (num_layers x batch_size x hidden_dim)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )
        # dim of H_o: 128 x 100 x 10 (batch_size x seq_len x hidden_dim)

        if D is not None or L is not None:
            C = torch.cat((D, L), 1)
            conditional_embedding = self.emb_conditional_linear(C)
            H_o = torch.cat((H_o, conditional_embedding.unsqueeze(1).repeat(1, H_o.shape[1], 1)), 2)
        if History is not None:
            H_o = torch.cat((H_o, History), 2)
        # 128 x 100 x 10
        logits = self.emb_linear(H_o)
        # dim of logits: 128 x 100 x 10 (batch_size x seq_len x hidden_dim)

        # 128 x 100 x 10
        # H = self.emb_sigmoid(logits)
        # dim of H: 128 x 100 x 10 (batch_size x seq_len x hidden_dim)

        return logits
