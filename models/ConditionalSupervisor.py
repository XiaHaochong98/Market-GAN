# -*- coding: UTF-8 -*-
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Blocks import *
from models.TimesNet import TimesNet


class Condition_supervisor(torch.nn.Module):
    # TCN network for condtion(district label) supervision
    def __init__(self, args):
        super(Condition_supervisor, self).__init__()
        model_list = {'TCN': TCNmodel, 'BiLSTM': BiLSTMClassifier, 'CNN': CNNClassifier, 'TimesNet': TimesNet,
                      'RNN': ConditionClassficiationRNN}
        print(f'Using {args.model} for condition supervision of task {args.task_name}')
        model_arg = copy.deepcopy(args)
        if args.task_name == 'latent_supervision':
            model_arg.task_name = 'classification'
        self.model = model_list[args.model](model_arg)
        self.args = args

    def forward(self, x, T=None, H=None):
        if self.args.task_name == 'latent_supervision' and self.args.model == 'RNN':
            x = self.model.forward(x, T, H)
        elif self.args.task_name == 'latent_supervision' and self.args.model == 'TimesNet':
            x = self.model.forward(x)
        else:
            x = self.model.forward(x)
        # get softmax of x
        # print('x.shape',x.shape)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class ConditionClassficiationRNN(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN
    """

    def __init__(self, args):
        super(ConditionClassficiationRNN, self).__init__()
        self.rnn_input_dim = args.hidden_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.args = args
        # Supervisor Architecture
        self.sup_rnn = torch.nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers - 1,
            batch_first=True
        )
        if args.add_history > 0:
            history_embedding_dim = args.hidden_size
        else:
            history_embedding_dim = 0
        self.sup_linear = torch.nn.Linear(self.hidden_dim + history_embedding_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()

        self.layer1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer2 = nn.Linear(args.hidden_size, args.num_class)
        # self.classification_linear=nn.Linear(args.hidden_size*args.max_seq_len,args.num_class)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.sup_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.sup_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T, History=None):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        # print('H',H.shape,'T',T.shape,'History',History.shape)
        H_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=H,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        # 128 x 100 x 10
        H_o, H_t = self.sup_rnn(H_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o,
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        if History is not None and self.args.add_history > 0:
            # conditional_embedding = self.sup_conditional_linear(History)
            # H_o = torch.cat((H_o, conditional_embedding.unsqueeze(1).repeat(1, H_o.shape[1], 1)), 2)
            H_o = torch.cat((H_o, History), 2)

        # 128 x 100 x 10
        H_hat = self.sup_linear(H_o)
        # 128 x 100 x 10
        B, T, C = H_hat.size()
        x = H_hat.view(B * T, C)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = x.view(B, T, -1)
        X = x.mean(dim=1)

        return X
