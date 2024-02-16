# We use the TimesNet one-step forcast model as the hiddenstate encoder

from argparse import Namespace

# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

from models.TimesNet import TimesNet


class HistoryEmbeddingNetwork(torch.nn.Module):
    def __init__(self, args):
        super(HistoryEmbeddingNetwork, self).__init__()
        self.args = args
        self.history_embedding_dim = args.hidden_size
        # update args
        HistoryEmbedding_args = Namespace(
            seq_len=args.history_length,
            task_name='encoding_to_hidden',
            pred_len=args.seq_len - args.history_length,
            c_out=self.history_embedding_dim,
            enc_in=args.feature_dim,
            condition_dim=0
        )
        # update self.args with HistoryEmbedding_args
        self.args.__dict__.update(HistoryEmbedding_args.__dict__)
        self.activation_fn = nn.Sigmoid()
        self.timesnet = TimesNet(self.args)

    def forward(self, X, T=None, D=None, L=None):
        X = self.timesnet(X, None)
        X = self.activation_fn(X)
        return X
