# add the project path to sys.path
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MarketGAN import MarketGAN
from models.models_utils import *
import torch
import numpy as np
from data.conditional_data_preprocess import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from argparse import Namespace


def configure_network_args(args):
    dynamic_supervisor_args = Namespace(
        model=args.pretrain_model,
        input_size=args.feature_dim,
        output_size=args.dynamic_dim,
        hidden_size=64,
        num_layers=2,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='classification',
        seq_len=args.max_seq_len,
        pred_len=0,
        e_layers=3,
        enc_in=args.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args.dynamic_dim,
    )
    label_supervisor_args = Namespace(
        model=args.pretrain_model,
        input_size=args.feature_dim,
        output_size=args.label_dim,
        hidden_size=64,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_layers=2,
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='classification',
        seq_len=args.max_seq_len,
        pred_len=0,
        e_layers=3,
        enc_in=args.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args.label_dim,
    )
    TimesNet_args = Namespace(
        use_TimesNet=args.use_TimesNet,
        use_RNN=args.use_RNN,
        add_history=args.add_history,
        input_size=args.feature_dim,
        output_size=args.label_dim,
        hidden_size=32,
        num_filters=64,
        filter_sizes=[2, 3, 4],
        num_layers=2,
        num_channels=[32],
        kernel_size=3,
        dropout=0.1,
        task_name='encoding',
        seq_len=args.max_seq_len,
        pred_len=0,
        e_layers=3,
        enc_in=args.feature_dim,
        hidden_dim=32,
        embed='timeF',
        freq='d',
        num_class=args.label_dim,
        feature_dim=args.feature_dim,
        condition_dim=args.dynamic_dim + args.label_dim
    )
    return dynamic_supervisor_args, label_supervisor_args, TimesNet_args


class MarketGAN_service():
    def __init__(self, model_path='../output/DJ30_V2_RT', device='cuda:0'):
        torch.autograd.set_detect_anomaly(True)
        self.model_path = model_path
        # variables = pd.read_pickle(f'{self.model_path}/preprocessed_data/variables.pkl')
        # create variables using the variables names in the variables list and read the data from '.output/preprocessed_data'
        # self.features=pd.read_pickle(f'{self.model_path}/features.pkl')
        # read
        with open(f'{self.model_path}/dynamics_tokenizer.pickle', 'rb') as f:
            self.dynamics_tokenizer = pickle.load(f)
        with open(f'{self.model_path}/tic_tokenizer.pickle', 'rb') as f:
            self.tic_tokenizer = pickle.load(f)
        with open(f'{self.model_path}/training_args.pickle', 'rb') as f:
            self.args = pickle.load(f)

        # torch.manual_seed(args.seed)
        # args.batch_size = 64
        # change device number here
        # args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(self.args)
        args = self.args
        args.device = device
        dynamic_supervisor_args, label_supervisor_args, TimesNet_args = configure_network_args(args)

        self.MarketGAN_instance = MarketGAN()
        data_path, args = self.MarketGAN_instance.init(args)
        print('MarketGAN_Service init done')

        # update the args
        self.MarketGAN_instance.dynamic_supervisor_args = dynamic_supervisor_args
        self.MarketGAN_instance.label_supervisor_args = label_supervisor_args
        self.MarketGAN_instance.TimesNet_args = TimesNet_args
        # generate the data
        args, model = self.MarketGAN_instance.load(args, self.model_path)
        print('MarketGAN_Service load done')
        self.model = model
        self.args = args

    def inference(self, dynamic, tic, H, scaler, Last_h, time):
        # we use the trained dynamics and label to test generation for now
        # Log start time
        model = self.model
        args = self.args

        tic_token = self.tic_tokenizer.word_to_one_hot(tic)
        # scaler_vector = [min_sclar_by_tic[tic], max_sclar_by_tic[tic]]
        tic_token = np.array(tic_token, dtype=float)
        tic_token = np.expand_dims(tic_token, axis=0)
        L = np.repeat(tic_token, H.shape[0], axis=0)  # dim: (n,tic_token)
        dynmamic_token = self.dynamics_tokenizer.word_to_one_hot(str(dynamic))
        dynmamic_token = np.array(dynmamic_token, dtype=float)
        dynmamic_token = np.expand_dims(dynmamic_token, axis=0)
        D = np.repeat(dynmamic_token, H.shape[0], axis=0)  # dim: (n,dynmamic_token)

        # if exceed_generation_length>0, do auto-regressive generation, otherwise, do normal generation
        # for auto-regressive generation, we need to use the last h as the input of the next time step
        # in this example we use the  train_H.shape[0] length slicing window of the train_H and generated data as train_H_slicing to get any length of generated data,
        # the step size is 1, and the value of each step is the mean value of all the real and generated data on that time step
        generated_X = conditional_timegan_generator(model, time, args, D, L, H)

        generated_time = time

        # rescale the generated data to the original scale
        # separate the train_Last_h to train_Last_h_data and train_Last_h_history on the last dimension
        Last_h_data, Last_h_history = np.split(Last_h, 2, axis=1)
        # separate the test_Last_h to test_Last_h_data and test_Last_h_history

        generated_X_rescaled = conditional_rescale_data(generated_X, scaler, args.differential_features, Last_h_data,
                                                        args.scaler_order, original_feature_order=args.feature)
        # X_rescaled = conditional_rescale_data(X, scaler, args.differential_features, Last_h_data, args.scaler_order, original_feature_order=args.feature)
        # history_rescaled = conditional_rescale_data(H, scaler, args.differential_features, Last_h_history, args.scaler_order, original_feature_order=args.feature)

        # generated_data_rescaled=np.concatenate((generated_data_as_train_rescaled,generated_data_as_test_rescaled),axis=0)

        return generated_X_rescaled
