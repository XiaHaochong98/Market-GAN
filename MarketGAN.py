# -*- coding: UTF-8 -*-
# Local modules
import random
import time
from argparse import Namespace

# 3rd-Party Modules
import numpy as np
import pandas as pd
import torch

from data.baseline_data_preprocess import data_preprocessing_for_condition_discrimination
from data.conditional_data_preprocess import *
from data.data_preprocess import *
from data.data_provider import *
from layers.Tokenizer import Tokenizer
from metrics.metric_utils import (
    one_step_ahead_prediction, post_hoc_discriminator, feature_constraint_evaluaton
)
from metrics.visualization import *
from models.conditional_timegan import Conditional_TimeGAN
from models.models_utils import *
from utils.stylized_facts_utils import *
from utils.util import *


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


def get_tics_PM(data_path):
    pass


class MarketGAN(object):
    def __init__(self):
        pass

    def init(self, args):
        ##############################################
        # Initialize output directories
        ##############################################
        ## Runtime directory
        code_dir = os.path.abspath(".")
        if not os.path.exists(code_dir):
            raise ValueError(f"Code directory not found at {code_dir}.")

        # Data directory
        # data_path = os.path.abspath("./data")
        data_path = args.data_path
        if not os.path.exists(data_path):
            raise ValueError(f"Data file not found at {data_path}.")
        data_dir = os.path.dirname(data_path)
        data_file_name = os.path.basename(data_path)

        # Output directories
        args.model_path = os.path.abspath(f"./output/{args.exp}/")
        out_dir = os.path.abspath(args.model_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        f = open(f"./output/{args.exp}/res.log", 'a')
        backup = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        # logging.basicConfig(filename="newfile.log",
        #                     format='%(asctime)s %(message)s',
        #                     filemode='w')
        # logger = logging.getLogger()
        # logger.setLevel(logging.INFO)

        # TensorBoard directory
        tensorboard_path = os.path.abspath("./tensorboard")
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path, exist_ok=True)

        # print(f"\nCode directory:\t\t\t{code_dir}")
        # print(f"Data directory:\t\t\t{data_path}")
        # print(f"Output directory:\t\t{out_dir}")
        # print(f"TensorBoard directory:\t\t{tensorboard_path}\n")

        ##############################################
        # Initialize random seed and CUDA
        ##############################################
        if args.seed == -1:
            args.seed = random.randint(1, 100000)

        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if args.device.startswith("cuda") and torch.cuda.is_available():
            print("Using CUDA\n")
            try:
                # os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
                args.device = torch.device(f"{args.device}")
                print(f"Using device {args.device}.\n")
            except:
                args.device = torch.device("cuda:0")
                print(f"Invalid device name {args.device}. Using cuda:0 instead.\n")
            # torch.cuda.manual_seed_all(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print("Using CPU\n")
            args.device = torch.device("cpu")
        return data_path, args

    def preprocess(self, data_path, args):
        #########################
        # Load and preprocess data for model
        #########################

        # choose the data preprocessing method based on the data structure
        if args.PM_data_structure == True:
            tics = get_tics_PM(data_path)
        else:
            print(data_path)
            tics = get_tics(data_path)
        # preprocess data with dynamic labels
        # tics=dataset_prepareation(data_path)

        # only take the first three tics for testing
        # tics=tics[:3]
        args.label_dim = len(tics)
        # one-hot tokenize tics and dynamic labels

        args.tics = tics
        # one hot tokenize tics
        label_book = tics
        tic_tokenizer = Tokenizer(num_words=len(label_book))
        tic_tokenizer.fit_on_text(label_book)

        # one hot tokenize dynamic labels
        dynamics_book = [str(i) for i in range(args.dynamic_dim)]
        dynamics_tokenizer = Tokenizer(num_words=args.dynamic_dim)
        dynamics_tokenizer.fit_on_text(dynamics_book)

        # dim of X (Idx x MaxSeqLen x Features)
        # dim of T (Idx)
        # dim of D (Idx x  one-hot dynamic labels)
        # dim of L (Idx x one-hot tics labels)
        # dim of scaler (Idx x (max,min))

        features = ['low', 'open', 'close', 'high']
        # store features into args
        args.feature = features
        print('using conditional data, waiting for preprocessing')
        if True or args.is_train == True:
            if args.conditional == True:
                X, H, T, D, L, scaler, scaler_order, Last_h, OX, OH = conditional_dataset_preprocessing(data_path, tics,
                                                                                                        args,
                                                                                                        tic_tokenizer,
                                                                                                        dynamics_tokenizer,
                                                                                                        features,
                                                                                                        history_length=args.history_length)

            else:
                X, H, T, D, L, scaler, scaler_order, Last_h, OX, OH = dataset_preprocessing(data_path, tics,
                                                                                            args, tic_tokenizer,
                                                                                            dynamics_tokenizer,
                                                                                            features,
                                                                                            history_length=args.history_length)
            args.padding_value = -1

            # store the scaler_order into args
            args.scaler_order = scaler_order

            print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
            print(f'Processed history data: {H.shape} (Idx x HistoryLe x Features)\n')
            print(f"Original data preview:\n{X[:2, :5, :1]}\n")
            print(f'Original history data preview:\n{H[:2, :5, :1]}\n')
            print(f"Original time preview:\n{T[:2]}\n")
            print(f"Original dynamic labels preview:\n{D[:2]}\n")
            print(f"Original tics labels preview:\n{L[:2]}\n")
            print(f"Original scaler preview:{scaler.shape}\n{scaler[:2]}\n")
            print(f'Last history data:\n{Last_h.shape}\n')

            args.feature_dim = X.shape[-1]
            args.dynamic_dim = D.shape[-1]
            args.label_dim = L.shape[-1]
            args.Z_dim = X.shape[-1]
            args.history_length = H.shape[1]

            # define dynamic supervisor args
            self.dynamic_supervisor_args = Namespace(
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
            self.label_supervisor_args = Namespace(
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
            self.TimesNet_args = Namespace(
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

            # Train-Test Split data and time
            train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler, train_H, test_H, train_Last_h, test_Last_h, OX_train, OX_test, OH_train, OH_test = train_test_split_by_conditions(
                args, dynamics_tokenizer, tic_tokenizer, X, T, D, L, scaler, H, Last_h, OX, OH)
            # print(f'train_X:{train_X.shape}')
            # print(f'test_X:{test_X.shape}')
            # print(f'train_time:{train_time.shape}')
            # print(f'test_time:{test_time.shape}')
            # print(f'train_D:{train_D.shape}')
            # print(f'test_D:{test_D.shape}')
            # print(f'train_L:{train_L.shape}')
            # print(f'test_L:{test_L.shape}')
            # print(f'train_scaler:{train_scaler.shape}')
            # print(f'test_scaler:{test_scaler.shape}')
            # print(f'train_H:{train_H.shape}')
            # print(f'test_H:{test_H.shape}')
            # print(f'train_Last_h:{train_Last_h.shape}')
            # print(f'test_Last_h:{test_Last_h.shape}')
            # print(f'OX_train:{OX_train.shape}')
            # print(f'OX_test:{OX_test.shape}')
            # print(f'OH_train:{OH_train.shape}')
            # print(f'OH_test:{OH_test.shape}')
            test_Last_h_data, test_Last_h_history = np.split(test_Last_h, 2, axis=1)
            if args.conditional:
                test_history_rescaled = conditional_rescale_data(test_H, test_scaler, args.differential_features,
                                                                 test_Last_h_history, args.scaler_order,
                                                                 original_feature_order=args.feature)
                test_data_rescaled = conditional_rescale_data(test_X, test_scaler, args.differential_features,
                                                              test_Last_h_data,
                                                              args.scaler_order, original_feature_order=args.feature)
            else:
                test_data_rescaled = rescale_data(test_X, test_scaler)
                test_history_rescaled = rescale_data(test_H, test_scaler)

            # check if the rescaled data is the same as the original data
            # if the element of OX is not none
            try:
                # rescale
                diff = test_data_rescaled - OX_test
                diff_h = test_history_rescaled - OH_test
                # diff_h=test_history_rescaled-orginal_test_history
                print("Diff between rescaled test data and original test data:", np.sum(diff))
                print("Diff between rescaled test history and original test history:", np.sum(diff_h))

                # process OX_test to encoding
                test_X_encode = data_preprocessing_for_condition_discrimination(test_history_rescaled,
                                                                                test_data_rescaled, test_Last_h_history,
                                                                                args.feature,
                                                                                args.differential_features)
                print('test_X_encode shape and preview:', test_X_encode.shape, test_X_encode[:1, :5, :])
                print('test_data_rescaled shape', test_data_rescaled.shape)
                print('test_X shape and preview:', test_X.shape, test_X[:1, :5, :])
                # diff between test_X_encode and test_X
                diff = test_X_encode - test_X
                # find the non-zero element of diff
                non_zero = np.nonzero(diff)
                print('number of non_zero element of diff:', len(non_zero[0]))
                print("Diff sum between test_X_encode and test_X:", np.sum(diff))
                # throw exception and end the program
            except:
                pass

            with open(f"{args.model_path}/train_X.pickle", "wb") as fb:
                pickle.dump(train_X, fb)
            with open(f"{args.model_path}/train_time.pickle", "wb") as fb:
                pickle.dump(train_time, fb)
            with open(f"{args.model_path}/test_X.pickle", "wb") as fb:
                pickle.dump(test_X, fb)
            with open(f"{args.model_path}/test_time.pickle", "wb") as fb:
                pickle.dump(test_time, fb)
            with open(f"{args.model_path}/train_D.pickle", "wb") as fb:
                pickle.dump(train_D, fb)
            with open(f"{args.model_path}/train_L.pickle", "wb") as fb:
                pickle.dump(train_L, fb)
            with open(f"{args.model_path}/test_D.pickle", "wb") as fb:
                pickle.dump(test_D, fb)
            with open(f"{args.model_path}/test_L.pickle", "wb") as fb:
                pickle.dump(test_L, fb)
            with open(f"{args.model_path}/train_scaler.pickle", "wb") as fb:
                pickle.dump(train_scaler, fb)
            with open(f"{args.model_path}/test_scaler.pickle", "wb") as fb:
                pickle.dump(test_scaler, fb)
            with open(f"{args.model_path}/train_H.pickle", "wb") as fb:
                pickle.dump(train_H, fb)
            with open(f"{args.model_path}/test_H.pickle", "wb") as fb:
                pickle.dump(test_H, fb)
            with open(f"{args.model_path}/train_Last_h.pickle", "wb") as fb:
                pickle.dump(train_Last_h, fb)
            with open(f"{args.model_path}/test_Last_h.pickle", "wb") as fb:
                pickle.dump(test_Last_h, fb)
        else:
            # load  train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler,train_H, test_H,train_Last_h,test_Last_h from file
            args.feature_dim = 4
            args.Z_dim = 4
            args.padding_value = -1
            args.scaler_order = ['low', 'O_minus_L', 'C_minus_L', 'H_minus_maxOC']
            # args.feature_dim = X.shape[-1]
            # args.dynamic_dim = D.shape[-1]
            # args.label_dim = L.shape[-1]
            self.dynamic_supervisor_args = Namespace(
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
            self.label_supervisor_args = Namespace(
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
            self.TimesNet_args = Namespace(
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

            # load train_X
            with open(f"{args.model_path}/train_X.pickle", "rb") as f:
                train_X = pickle.load(f)
            # load test_X
            with open(f"{args.model_path}/test_X.pickle", "rb") as f:
                test_X = pickle.load(f)
            # load train_time
            with open(f"{args.model_path}/train_time.pickle", "rb") as f:
                train_time = pickle.load(f)
            # load test_time
            with open(f"{args.model_path}/test_time.pickle", "rb") as f:
                test_time = pickle.load(f)
            # load train_D
            with open(f"{args.model_path}/train_D.pickle", "rb") as f:
                train_D = pickle.load(f)
            # load test_D
            with open(f"{args.model_path}/test_D.pickle", "rb") as f:
                test_D = pickle.load(f)
            # load train_L
            with open(f"{args.model_path}/train_L.pickle", "rb") as f:
                train_L = pickle.load(f)
            # load test_L
            with open(f"{args.model_path}/test_L.pickle", "rb") as f:
                test_L = pickle.load(f)
            # load train_scaler
            with open(f"{args.model_path}/train_scaler.pickle", "rb") as f:
                train_scaler = pickle.load(f)
            # load test_scaler
            with open(f"{args.model_path}/test_scaler.pickle", "rb") as f:
                test_scaler = pickle.load(f)
            # load train_H
            with open(f"{args.model_path}/train_H.pickle", "rb") as f:
                train_H = pickle.load(f)
            # load test_H
            with open(f"{args.model_path}/test_H.pickle", "rb") as f:
                test_H = pickle.load(f)
            # load train_Last_h
            with open(f"{args.model_path}/train_Last_h.pickle", "rb") as f:
                train_Last_h = pickle.load(f)
            # load test_Last_h
            with open(f"{args.model_path}/test_Last_h.pickle", "rb") as f:
                test_Last_h = pickle.load(f)
            print('load data from files')

        return args, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler, train_H, test_H, train_Last_h, test_Last_h, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer, features

    def load(self, args, model_path=None):
        # load model
        TimesNet_args = self.TimesNet_args
        label_supervisor_args = self.label_supervisor_args
        dynamic_supervisor_args = self.dynamic_supervisor_args
        args.__dict__.update(TimesNet_args.__dict__)
        # print('TimesNet_args',TimesNet_args)
        # print('label_supervisor_args',label_supervisor_args)
        # print('dynamic_supervisor_args',dynamic_supervisor_args)
        model = Conditional_TimeGAN(args, dynamic_supervisor_args=dynamic_supervisor_args,
                                    label_supervisor_args=label_supervisor_args)
        if model_path is None:
            model_path = args.model_path
        print(f"loading model from {model_path}/model.pt")
        try:
            model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=torch.device(args.device)))
        except Exception as error:
            # print loading error
            print('loading model error:' + repr(error))
        model = model.to(args.device)
        print('model on device:' + str(next(model.parameters()).device))
        return args, model

    def export_exp_info(self, data_path, args):
        # choose the data preprocessing method based on the data structure
        if args.PM_data_structure == True:
            tics = get_tics_PM(data_path)
        else:
            print(data_path)
            tics = get_tics(data_path)
        args.label_dim = len(tics)
        # one-hot tokenize tics and dynamic labels

        args.tics = tics
        # one hot tokenize tics
        label_book = tics
        tic_tokenizer = Tokenizer(num_words=len(label_book))
        tic_tokenizer.fit_on_text(label_book)

        # one hot tokenize dynamic labels
        dynamics_book = [str(i) for i in range(args.dynamic_dim)]
        dynamics_tokenizer = Tokenizer(num_words=args.dynamic_dim)
        dynamics_tokenizer.fit_on_text(dynamics_book)

        # dim of X (Idx x MaxSeqLen x Features)
        # dim of T (Idx)
        # dim of D (Idx x  one-hot dynamic labels)
        # dim of L (Idx x one-hot tics labels)
        # dim of scaler (Idx x (max,min))

        features = ['low', 'open', 'close', 'high']

        args.feature = features

        X, H, T, D, L, scaler, scaler_order, Last_h, OX, OH = conditional_dataset_preprocessing(data_path, tics,
                                                                                                args, tic_tokenizer,
                                                                                                dynamics_tokenizer,
                                                                                                features,
                                                                                                history_length=args.history_length)

        args.padding_value = -1

        # store the scaler_order into args
        args.scaler_order = scaler_order

        print(f"Processed data: {X.shape} (Idx x MaxSeqLen x Features)\n")
        print(f'Processed history data: {H.shape} (Idx x HistoryLe x Features)\n')
        print(f"Original data preview:\n{X[:2, :5, :1]}\n")
        print(f'Original history data preview:\n{H[:2, :5, :1]}\n')
        print(f"Original time preview:\n{T[:2]}\n")
        print(f"Original dynamic labels preview:\n{D[:2]}\n")
        print(f"Original tics labels preview:\n{L[:2]}\n")
        print(f"Original scaler preview:{scaler.shape}\n{scaler[:2]}\n")
        print(f'Last history data:\n{Last_h.shape}\n')

        args.feature_dim = X.shape[-1]
        args.dynamic_dim = D.shape[-1]
        args.label_dim = L.shape[-1]
        args.Z_dim = X.shape[-1]
        args.history_length = H.shape[1]

        # save tic_tokenizer and dynamics_tokenizer and features to pickle file
        with open(f"{args.model_path}/tic_tokenizer.pickle", "wb") as fb:
            pickle.dump(tic_tokenizer, fb)
        with open(f"{args.model_path}/dynamics_tokenizer.pickle", "wb") as fb:
            pickle.dump(dynamics_tokenizer, fb)
        with open(f"{args.model_path}/features.pickle", "wb") as fb:
            pickle.dump(features, fb)
        with open(f"{args.model_path}/training_args.pickle", "wb") as fb:
            pickle.dump(args, fb)

    def data_provider(self, data_path, args):
        #########################
        # Load and preprocess data for model
        #########################

        # choose the data preprocessing method based on the data structure
        if args.PM_data_structure == True:
            tics = get_tics_PM(data_path)
        else:
            print(data_path)
            tics = get_tics(data_path)
        args.label_dim = len(tics)
        # one-hot tokenize tics and dynamic labels

        args.tics = tics
        # one hot tokenize tics
        label_book = tics
        tic_tokenizer = Tokenizer(num_words=len(label_book))
        tic_tokenizer.fit_on_text(label_book)

        # one hot tokenize dynamic labels
        dynamics_book = [str(i) for i in range(args.dynamic_dim)]
        dynamics_tokenizer = Tokenizer(num_words=args.dynamic_dim)
        dynamics_tokenizer.fit_on_text(dynamics_book)

        # dim of X (Idx x MaxSeqLen x Features)
        # dim of T (Idx)
        # dim of D (Idx x  one-hot dynamic labels)
        # dim of L (Idx x one-hot tics labels)
        # dim of scaler (Idx x (max,min))

        features = ['low', 'open', 'close', 'high']
        # store features into args
        args.feature = features

        if not args.prepare_encoded_data:
            print('preparing original data samples')
            data, D, L, H, Last_h = \
                data_provider(data_path, tics,
                              args, tic_tokenizer,
                              dynamics_tokenizer,
                              features,
                              history_length=args.history_length)
            # creat dummy T, scaler, H, Last_h, OX, OH which have the same shape as X
            T = np.zeros(data.shape[0])
            scaler = np.zeros((data.shape[0], 2))
        else:
            print('preparing encoded data samples')
            data, H, T, D, L, scaler, scaler_order, Last_h, OX, OH = conditional_dataset_preprocessing(data_path, tics,
                                                                                                       args,
                                                                                                       tic_tokenizer,
                                                                                                       dynamics_tokenizer,
                                                                                                       features,
                                                                                                       history_length=args.history_length)
            # we need to concate the H before the data in each sample to the data in each sample
            data = np.concatenate((H, data), axis=1)
            # testify is the first args.history_length rows of data is the same as H
            testify = np.all(data[:, :args.history_length, :] == H, axis=(1, 2))
            # throw exception and end the program if testify is not true
            if not testify.all():
                print('H is concatenate in the wrong way')
                raise Exception
            else:
                print('H is concatenate in the right way')

        OX = np.zeros((data.shape[0], args.history_length, len(features)))
        OH = np.zeros((data.shape[0], args.history_length, len(features)))
        train_data, test_data, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler, train_H, test_H, train_Last_h, test_Last_h, OX_train, OX_test, OH_train, OH_test = train_test_split_by_conditions(
            args, dynamics_tokenizer, tic_tokenizer, data, T, D, L, scaler, H, Last_h, OX, OH)
        # train_X, test_X, train_D, test_D, train_L, test_L = train_test_split(
        #     X, D, L, test_size=1-args.train_rate, random_state=args.seed
        # )
        for dynamic in range(args.dynamic_dim):
            for tic in args.tics:
                print('train dynamic {} tic {}'.format(dynamic, tic))

                train_D_word = dynamics_tokenizer.one_hot_to_word(train_D)
                train_L_word = tic_tokenizer.one_hot_to_word(train_L)
                test_D_word = dynamics_tokenizer.one_hot_to_word(test_D)
                test_L_word = tic_tokenizer.one_hot_to_word(test_L)

                # find the index of the data where train_D==dynamic and train_L==tic from train_D_word and train_L_word(lists)
                train_index = []
                for i in range(len(train_D_word)):
                    if train_D_word[i] == str(dynamic) and train_L_word[i] == str(tic):
                        train_index.append(i)
                train_index = np.array(train_index)
                # find the index of the data where train_D==dynamic and train_L==tic from test_D_word and test_L_word(lists)
                test_index = []
                for i in range(len(test_D_word)):
                    if test_D_word[i] == str(dynamic) and test_L_word[i] == str(tic):
                        test_index.append(i)
                test_index = np.array(test_index)
                train_data_slice = train_data[train_index]
                test_data_slice = test_data[test_index]
                print(f'train data slice shape of {tic} and {dynamic}: {train_data_slice.shape}')
                print(f'test data slice shape of {tic} and {dynamic}: {test_data_slice.shape}')
                # save data slice to npy file
                np.save(f'{args.model_path}/{tic}_{dynamic}_train.npy', train_data_slice)
                np.save(f'{args.model_path}/{tic}_{dynamic}_test.npy', test_data_slice)

        # save ,dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer,train_X, test_X,
        #                              train_time, train_D, train_L, train_H, test_time, test_D, test_L, test_H,train_scaler, test_scaler, train_Last_h, test_Last_h
        # to args.model_path for later use as pickle file in a dictionary with the variable name as key
        # create a dictionary to store all the variables
        data_provider_dict = {}
        data_provider_dict['dynamics_book'] = dynamics_book
        data_provider_dict['dynamics_tokenizer'] = dynamics_tokenizer
        data_provider_dict['label_book'] = label_book
        data_provider_dict['tic_tokenizer'] = tic_tokenizer
        data_provider_dict['train_data'] = train_data
        data_provider_dict['test_data'] = test_data
        data_provider_dict['train_time'] = train_time
        data_provider_dict['train_D'] = train_D
        data_provider_dict['train_L'] = train_L
        data_provider_dict['train_H'] = train_H
        data_provider_dict['test_time'] = test_time
        data_provider_dict['test_D'] = test_D
        data_provider_dict['test_L'] = test_L
        data_provider_dict['test_H'] = test_H
        data_provider_dict['train_Last_h'] = train_Last_h
        data_provider_dict['test_Last_h'] = test_Last_h
        data_provider_dict['train_scaler'] = train_scaler
        data_provider_dict['test_scaler'] = test_scaler
        print('train time shape: ', train_time.shape)
        print('train D shape: ', train_D.shape)
        print('train L shape: ', train_L.shape)
        print('train H shape: ', train_H.shape)
        print('test time shape: ', test_time.shape)
        print('test D shape: ', test_D.shape)
        print('test L shape: ', test_L.shape)
        print('test H shape: ', test_H.shape)
        print('train Last h shape: ', train_Last_h.shape)
        print('test Last h shape: ', test_Last_h.shape)
        print('train scaler shape: ', train_scaler.shape)
        print('test scaler shape: ', test_scaler.shape)
        with open(f'{args.model_path}/data_provider.pkl', 'wb') as f:
            pickle.dump(data_provider_dict, f)
        print('variables saved to pickle file')
        print('data provider finished')

    def load_model(self, args):

        TimesNet_args = self.TimesNet_args
        label_supervisor_args = self.label_supervisor_args
        dynamic_supervisor_args = self.dynamic_supervisor_args
        # merge with args TimesNet_args
        args.__dict__.update(TimesNet_args.__dict__)

        print("Loading model...\n")
        print('Mode: conditional')
        model = Conditional_TimeGAN(args, dynamic_supervisor_args=dynamic_supervisor_args,
                                    label_supervisor_args=label_supervisor_args)
        try:
            model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
        except Exception as error:
            # print loading error
            print('loading model error:' + repr(error))
            # sys.exit(0)
            # return 0
        model = model.to(args.device)
        print('model on device:' + str(next(model.parameters()).device))
        return args, model

    def train(self, args, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler,
              test_scaler, train_H, test_H, train_Last_h, test_Last_h):
        #########################
        # Initialize and Run model
        #########################
        # log args
        print("Experiment parameters:", args, "\n")

        start_time = time.time()

        TimesNet_args = self.TimesNet_args
        label_supervisor_args = self.label_supervisor_args
        dynamic_supervisor_args = self.dynamic_supervisor_args
        # merge with args TimesNet_args
        args.__dict__.update(TimesNet_args.__dict__)

        if args.is_train == True:
            print("Training model...\n")
            if args.conditional:
                print('Mode: conditional')
            else:
                print('Mode: original')
            model = Conditional_TimeGAN(args, dynamic_supervisor_args=dynamic_supervisor_args,
                                        label_supervisor_args=label_supervisor_args)
            conditional_timegan_trainer(model, train_X, train_time, args, train_D, train_L, train_H)


        else:
            print("Loading model...\n")
            if args.conditional:
                print('Mode: conditional')
                model = Conditional_TimeGAN(args, dynamic_supervisor_args=dynamic_supervisor_args,
                                            label_supervisor_args=label_supervisor_args)
                try:
                    model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
                except Exception as error:
                    # print loading error
                    print('loading model error:' + repr(error))
                model = model.to(args.device)
                print('model on device:' + str(next(model.parameters()).device))
            else:
                print('Mode: original')
                # model = TimeGAN(args)
                model = Conditional_TimeGAN(args, dynamic_supervisor_args=dynamic_supervisor_args,
                                            label_supervisor_args=label_supervisor_args)
                model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
                model = model.to(args.device)
                print('model on device:' + str(next(model.parameters()).device))

        # log training time start -end
        end_time = time.time()
        print('Training time: ' + str(end_time - start_time))
        return args, model

    def generation(self, args, model, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L,
                   train_scaler,
                   test_scaler, train_H, test_H, train_Last_h, test_Last_h, exceed_generation_length=0):
        # we use the trained dynamics and label to test generation for now
        # Log start time
        evaluation_test = False
        print(f'Using randoms noise as generated data: {evaluation_test}')
        start_time = time.time()
        if args.conditional:
            print('conditional mode generation')

            # if exceed_generation_length>0, do auto-regressive generation, otherwise, do normal generation
            # for auto-regressive generation, we need to use the last h as the input of the next time step
            # in this example we use the  train_H.shape[0] length slicing window of the train_H and generated data as train_H_slicing to get any length of generated data,
            # the step size is 1, and the value of each step is the mean value of all the real and generated data on that time step
            print('args1', args, args.batch_size)
            generated_test_X = conditional_timegan_generator(model, test_time, args, test_D, test_L, test_H)
            generated_train_X = conditional_timegan_generator(model, train_time, args, train_D, train_L, train_H)
        else:
            print('original mode generation')
            generated_test_X = conditional_timegan_generator(model, test_time, args, None, None, None)
            generated_train_X = conditional_timegan_generator(model, train_time, args, None, None, None)
        if evaluation_test:
            print('testing evaluation pipeline,using random nosie as generated data')
            generated_test_X = np.random.uniform(0, 1, size=(test_X.shape[0], test_X.shape[1], test_X.shape[2]))
            generated_train_X = np.random.uniform(0, 1, size=(train_X.shape[0], train_X.shape[1], train_X.shape[2]))
        generated_time_as_test = test_time
        generated_time_as_train = train_time

        # Log end time
        end_time = time.time()

        # rescale the generated data to the original scale
        # sperate the train_Last_h to train_Last_h_data and train_Last_h_history on the last dimension
        train_Last_h_data, train_Last_h_history = np.split(train_Last_h, 2, axis=1)
        test_Last_h_data, test_Last_h_history = np.split(test_Last_h, 2, axis=1)
        # sperate the test_Last_h to test_Last_h_data and test_Last_h_history

        if args.conditional:
            generated_data_as_train_rescaled = conditional_rescale_data(generated_train_X, train_scaler,
                                                                        args.differential_features, train_Last_h_data,
                                                                        args.scaler_order,
                                                                        original_feature_order=args.feature)
            generated_data_as_test_rescaled = conditional_rescale_data(generated_test_X, test_scaler,
                                                                       args.differential_features, test_Last_h_data,
                                                                       args.scaler_order,
                                                                       original_feature_order=args.feature)
            # print('rescale train')
            train_data_rescaled = conditional_rescale_data(train_X, train_scaler, args.differential_features,
                                                           train_Last_h_data, args.scaler_order,
                                                           original_feature_order=args.feature)
            test_data_rescaled = conditional_rescale_data(test_X, test_scaler, args.differential_features,
                                                          test_Last_h_data, args.scaler_order,
                                                          original_feature_order=args.feature)
            train_history_rescaled = conditional_rescale_data(train_H, train_scaler, args.differential_features,
                                                              train_Last_h_history, args.scaler_order,
                                                              original_feature_order=args.feature)
            test_history_rescaled = conditional_rescale_data(test_H, test_scaler, args.differential_features,
                                                             test_Last_h_history, args.scaler_order,
                                                             original_feature_order=args.feature)
        else:
            generated_data_as_train_rescaled = rescale_data(generated_train_X, train_scaler)
            generated_data_as_test_rescaled = rescale_data(generated_test_X, test_scaler)
            train_data_rescaled = rescale_data(train_X, train_scaler)
            test_data_rescaled = rescale_data(test_X, test_scaler)
            train_history_rescaled = rescale_data(train_H, train_scaler)
            test_history_rescaled = rescale_data(test_H, test_scaler)
        print(
            f'Generated data as train rescaled preview:\n{generated_data_as_train_rescaled.shape, generated_data_as_train_rescaled[:1, :5, :2]}\n')
        print(f'Train data rescaled preview:\n{train_data_rescaled.shape, train_data_rescaled[:1, :5, :2]}\n')
        print(
            f'Generated data as history rescaled preview:\n{train_history_rescaled.shape, train_history_rescaled[:1, :5, :2]}\n')
        print(f'Train history rescaled preview:\n{train_history_rescaled.shape, train_history_rescaled[:1, :5, :2]}\n')
        print(f"Model Generation Runtime: {(end_time - start_time) / 60} mins\n")

        generated_data_rescaled = np.concatenate((generated_data_as_train_rescaled, generated_data_as_test_rescaled),
                                                 axis=0)
        #########################
        # Save train and generated data for visualization
        #########################

        # Save generated data

        with open(f"{args.model_path}/fake_data_train.pickle", "wb") as fb:
            pickle.dump(generated_train_X, fb)
        with open(f"{args.model_path}/fake_time_train.pickle", "wb") as fb:
            pickle.dump(generated_time_as_train, fb)
        with open(f"{args.model_path}/fake_data_test.pickle", "wb") as fb:
            pickle.dump(generated_test_X, fb)
        with open(f"{args.model_path}/fake_time_test.pickle", "wb") as fb:
            pickle.dump(generated_time_as_test, fb)
        with open(f"{args.model_path}/fake_data_rescaled.pickle", "wb") as fb:
            pickle.dump(generated_data_rescaled, fb)

        return args, model, generated_train_X, generated_test_X, train_X, test_X, generated_data_as_train_rescaled, generated_data_as_test_rescaled, train_data_rescaled, test_data_rescaled, train_history_rescaled, test_history_rescaled, generated_time_as_test, generated_time_as_train

    def evaluate(self, args, model, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer,
                 generated_train_X, generated_test_X, train_X, test_X,
                 generated_data_as_train_rescaled, generated_data_as_test_rescaled, train_data_rescaled,
                 test_data_rescaled,
                 train_history_rescaled, test_history_rescaled,
                 train_Last_h, test_Last_h,
                 train_time, train_D, train_L, train_H,
                 test_time, test_D, test_L, test_H,
                 exp_name, result_path=None, model_name=''):

        # TSNE plot of generated data and test data
        # random pick data of specific condition to plot
        # dynamics
        plot = args.plot
        if plot:
            data_list = []
            generated_data_list = []
            norm_data_list = []
            norm_generated_data_list = []
            data_list_t = []
            generated_data_list_t = []
            norm_data_list_t = []
            norm_generated_data_list_t = []

            def normalize_data(fake_data, real_data):
                # normalize the data with the Standard scaler fit on real data of each sample
                # fake_data shape: (sample_size, time_step, feature_size)
                # real_data shape: (sample_size, time_step, feature_size)
                # return fake_data,real_data
                scaler_list = []
                # creat placeholder norm_fake_data and norm_real_data
                norm_fake_data = np.zeros(fake_data.shape)
                norm_real_data = np.zeros(real_data.shape)
                for i in range(fake_data.shape[0]):
                    scaler = StandardScaler()
                    scaler.fit(real_data[i])
                    scaler_list.append(scaler)
                for i in range(fake_data.shape[0]):
                    norm_fake_data[i] = scaler_list[i].transform(fake_data[i])
                    norm_real_data[i] = scaler_list[i].transform(real_data[i])
                return norm_fake_data, norm_real_data

            generated_data_as_train_rescaled_norm, train_data_rescaled_norm = normalize_data(
                generated_data_as_train_rescaled, train_data_rescaled)
            visualization(generated_data_as_train_rescaled, train_data_rescaled, analysis="tsne", args=args,
                          fig_suffix=f'{model_name}_train')
            generated_data_as_test_rescaled_norm, test_data_rescaled_norm = normalize_data(
                generated_data_as_test_rescaled, test_data_rescaled)
            visualization(generated_data_as_test_rescaled, test_data_rescaled, analysis="tsne", args=args,
                          fig_suffix=f'{model_name}_test')
            for d in dynamics_book:
                # get the token of the dynamics
                d_token = dynamics_tokenizer.word_to_one_hot(d)

                d_index = np.where((train_D == d_token).all(axis=1))[0]
                d_index_t = np.where((test_D == d_token).all(axis=1))[0]
                print(f'dynamics data size of {d} is {len(d_index)} out of {train_data_rescaled.shape[0]}')
                # get the data of the d_token index in the train_X
                real_d_data = train_data_rescaled[d_index]
                real_d_data_t = test_data_rescaled[d_index_t]
                # get the data of the d_token index in the generated_data
                fake_d_data = generated_data_as_train_rescaled[d_index]
                fake_d_data_t = generated_data_as_test_rescaled[d_index_t]
                # normalize the fake_d_data and real_d_data with the StandardScaler of the real_d_data of each sample without creating another function
                fake_d_data_norm, real_d_data_norm = normalize_data(fake_d_data, real_d_data)
                fake_d_data_norm_t, real_d_data_norm_t = normalize_data(fake_d_data_t, real_d_data_t)

                data_list.append(real_d_data)
                generated_data_list.append(fake_d_data)
                norm_data_list.append(real_d_data_norm)
                norm_generated_data_list.append(fake_d_data_norm)
                data_list_t.append(real_d_data_t)
                generated_data_list_t.append(fake_d_data_t)
                norm_data_list_t.append(real_d_data_norm_t)
                norm_generated_data_list_t.append(fake_d_data_norm_t)
                visualization(fake_d_data, real_d_data, analysis="tsne", args=args,
                              fig_suffix=f'{model_name}_train_dynamics_{d}')
                visualization(fake_d_data_t, real_d_data_t, analysis="tsne", args=args,
                              fig_suffix=f'{model_name}_test_dynamics_{d}')
            visualization_by_dynamic_solo(norm_data_list[0], norm_data_list[1], norm_data_list[2], args=args,
                                          fig_suffix=f'Real_dynamics_plot_norm_train')
            visualization_by_dynamic_solo(norm_data_list_t[0], norm_data_list_t[1], norm_data_list_t[2], args=args,
                                          fig_suffix=f'Real_dynamics_plot_norm_test')
            visualization_by_dynamic_solo(norm_generated_data_list[0], norm_generated_data_list[1],
                                          norm_generated_data_list[2], args=args,
                                          fig_suffix=f'{model_name}_dynamics_plot_norm_train')
            visualization_by_dynamic_solo(norm_generated_data_list_t[0], norm_generated_data_list_t[1],
                                          norm_generated_data_list_t[2], args=args,
                                          fig_suffix=f'{model_name}_dynamics_plot_norm_test')
            # # labels
            # # randomly pick 3 labels from the label_book
            # # label_book_random_picked = np.random.choice(label_book, min(3,len(label_book)), replace=False)
            label_book_random_picked = label_book[:10]
            for l in label_book_random_picked:
                # get the token of the label
                l_token = tic_tokenizer.word_to_one_hot(l)
                # find the index of the l_token in the train_L
                l_index = np.where((train_L == l_token).all(axis=1))[0]
                l_index_t = np.where((test_L == l_token).all(axis=1))[0]
                print(f'label data size of {l} is {len(l_index)} out of {train_data_rescaled.shape[0]}')
                # get the data of the l_token index
                real_l_data_t = test_data_rescaled[l_index_t]
                # # get the data of the l_token index in the generated_data
                fake_l_data_t = generated_data_as_train_rescaled[l_index_t]

                fake_l_data_norm_t, real_l_data_norm_t = normalize_data(fake_l_data_t, real_l_data_t)
                # visualize the data
                print(f"Running TSNE plot of generated data and real data for label {l} {model_name} on test set...")
                visualization(fake_l_data_t, real_l_data_t, analysis="tsne", args=args,
                              fig_suffix=f'{model_name}_test_label_{l}')
            return 0
        #########################
        # Evaluate the performance
        #########################
        suffix = ''
        print('expriement suffix:', suffix)
        evaluation_epochs = args.evalution_epochs
        print('evaluation_epochs:', evaluation_epochs)
        # Evaluate the ability of evaluation model to distinguish the random noise and the real data
        # replace the generated data with random noise that have same shape as generated data,each element is sampled from uniform distribution that have the same(mean,std) as the generated data

        # record all the evaluation results in a dictionary and write to a csv file
        evaluation_results = {}
        evaluation_results_high_level = {}
        print('generated_data_as_train_rescaled preview', generated_data_as_train_rescaled[:1, :5, :])
        print('train_data_rescaled preview', train_data_rescaled[:1, :5, :])
        print('generated_data_as_test_rescaled preview', generated_data_as_test_rescaled[:1, :5, :])
        print('test_data_rescaled preview', test_data_rescaled[:1, :5, :])
        train_Last_h_data, train_Last_h_history = np.split(train_Last_h, 2, axis=1)
        test_Last_h_data, test_Last_h_history = np.split(test_Last_h, 2, axis=1)

        original_data_rescaled = np.concatenate((train_data_rescaled, test_data_rescaled), axis=0)
        generated_time_as_train = train_time
        features = args.feature

        discrimination_loss_train_list = []
        discrimination_loss_test_list = []
        discrimination_acc_train_list = []
        discrimination_acc_test_list = []
        real_step_ahead_pred_SMAPE_list = []
        real_step_ahead_pred_SMAPE_norm_list = []
        real_step_ahead_pred_SMAPE_avg_list = []
        real_step_ahead_pred_training_MSE_list = []
        real_fake_step_ahead_pred_SMAPE_list = []
        real_fake_step_ahead_pred_SMAPE_norm_list = []
        real_fake_step_ahead_pred_SMAPE_avg_list = []
        real_fake_step_ahead_pred_training_MSE_list = []
        train_D_word = dynamics_tokenizer.one_hot_to_word(train_D)
        train_L_word = tic_tokenizer.one_hot_to_word(train_L)
        test_D_word = dynamics_tokenizer.one_hot_to_word(test_D)
        test_L_word = tic_tokenizer.one_hot_to_word(test_L)
        # #
        for dynamic in range(args.dynamic_dim):
            for tic in args.tics:
                print('discrimator  dynamic {} tic {}'.format(dynamic, tic))
                # slice the data for each dynamic and tic

                # find the index of the data where train_D==dynamic and train_L==tic from train_D_word and train_L_word(lists)
                train_index = []
                for i in range(len(train_D_word)):
                    if train_D_word[i] == str(dynamic) and train_L_word[i] == str(tic):
                        train_index.append(i)
                train_index = np.array(train_index)
                # find the index of the data where train_D==dynamic and train_L==tic from test_D_word and test_L_word(lists)
                test_index = []
                for i in range(len(test_D_word)):
                    if test_D_word[i] == str(dynamic) and test_L_word[i] == str(tic):
                        test_index.append(i)
                test_index = np.array(test_index)
                # slice the data for each dynamic and tic
                # train
                train_data_rescaled_dynamic_tic = train_data_rescaled[train_index]
                train_time_dynamic_tic = train_time[train_index]
                train_Last_h_data_dynamic_tic = train_Last_h_data[train_index]
                train_Last_h_history_dynamic_tic = train_Last_h_history[train_index]
                # test
                test_data_rescaled_dynamic_tic = test_data_rescaled[test_index]
                test_time_dynamic_tic = test_time[test_index]
                test_Last_h_data_dynamic_tic = test_Last_h_data[test_index]
                test_Last_h_history_dynamic_tic = test_Last_h_history[test_index]
                # train the discriminator
                generated_data_as_train_rescaled_dynamic_tic = generated_data_as_train_rescaled[train_index]
                generated_data_as_test_rescaled_dynamic_tic = generated_data_as_test_rescaled[test_index]
                discriminator_acc, discriminator_loss = post_hoc_discriminator(
                    (test_data_rescaled_dynamic_tic, test_time_dynamic_tic),
                    (generated_data_as_test_rescaled_dynamic_tic, test_time_dynamic_tic),
                    args,
                    epoch=evaluation_epochs
                )
                discriminator_loss = discriminator_loss.item()
                discrimination_loss_test = abs(discriminator_acc - 50)
                discrimination_acc_test_list.append(discriminator_acc)
                discrimination_loss_test_list.append(discrimination_loss_test)

                # One step ahead prediction: TSTR
                # test data augmentation ability of the generator
                print("TSTR:Running one step ahead prediction using original data as training set...")
                print('train_data_rescaled_dynamic_tic_1', train_data_rescaled_dynamic_tic[0, :2, :2])
                real_step_ahead_pred_SMAPE, real_step_ahead_pred_SMAPE_norm, real_step_ahead_pred_training_MSE = one_step_ahead_prediction(
                    (train_data_rescaled_dynamic_tic, train_time_dynamic_tic),
                    (test_data_rescaled_dynamic_tic, test_time_dynamic_tic),
                    args,
                    epochs=evaluation_epochs
                )
                real_step_ahead_pred_SMAPE_avg = (real_step_ahead_pred_SMAPE + real_step_ahead_pred_SMAPE_norm) / 2
                print('TSTR:Running one step ahead prediction using original+generated data as training set...')
                print('train_data_rescaled_dynamic_tic_2', train_data_rescaled_dynamic_tic[0, :2, :2])
                real_fake_step_ahead_pred_SMAPE, real_fake_step_ahead_pred_SMAPE_norm, real_fake_step_ahead_pred_training_MSE = one_step_ahead_prediction(
                    (np.concatenate((train_data_rescaled_dynamic_tic, generated_data_as_train_rescaled_dynamic_tic),
                                    axis=0),
                     np.concatenate((train_time_dynamic_tic, train_time_dynamic_tic), axis=0)),
                    (test_data_rescaled_dynamic_tic, test_time_dynamic_tic),
                    args,
                    epochs=max(evaluation_epochs // 2, 1)
                )
                real_fake_step_ahead_pred_SMAPE_avg = (
                                                              real_fake_step_ahead_pred_SMAPE + real_fake_step_ahead_pred_SMAPE_norm) / 2
                real_step_ahead_pred_SMAPE_list.append(real_step_ahead_pred_SMAPE)
                real_step_ahead_pred_SMAPE_norm_list.append(real_step_ahead_pred_SMAPE_norm)
                real_step_ahead_pred_SMAPE_avg_list.append(real_step_ahead_pred_SMAPE_avg)
                real_step_ahead_pred_training_MSE_list.append(real_step_ahead_pred_training_MSE)
                real_fake_step_ahead_pred_SMAPE_list.append(real_fake_step_ahead_pred_SMAPE)
                real_fake_step_ahead_pred_SMAPE_norm_list.append(real_fake_step_ahead_pred_SMAPE_norm)
                real_fake_step_ahead_pred_SMAPE_avg_list.append(real_fake_step_ahead_pred_SMAPE_avg)
                real_fake_step_ahead_pred_training_MSE_list.append(real_fake_step_ahead_pred_training_MSE)
        evaluation_results['real_step_ahead_pred_SMAPE'] = sum(real_step_ahead_pred_SMAPE_list) / len(
            real_step_ahead_pred_SMAPE_list)
        evaluation_results['real_step_ahead_pred_SMAPE_norm'] = sum(real_step_ahead_pred_SMAPE_norm_list) / len(
            real_step_ahead_pred_SMAPE_norm_list)
        evaluation_results['real_step_ahead_pred_training_MSE'] = sum(real_step_ahead_pred_training_MSE_list) / len(
            real_step_ahead_pred_training_MSE_list)
        evaluation_results['real_step_ahead_pred_SMAPE_avg'] = sum(real_step_ahead_pred_SMAPE_avg_list) / len(
            real_step_ahead_pred_SMAPE_avg_list)
        evaluation_results['real_step_ahead_pred_SMAPE_std'] = np.std(real_step_ahead_pred_SMAPE_list)
        evaluation_results['real_step_ahead_pred_SMAPE_norm_std'] = np.std(real_step_ahead_pred_SMAPE_norm_list)
        evaluation_results['real_step_ahead_pred_training_MSE_std'] = np.std(real_step_ahead_pred_training_MSE_list)
        evaluation_results['real_step_ahead_pred_SMAPE_avg_std'] = np.std(real_step_ahead_pred_SMAPE_avg_list)
        #     # store the SMAPE as str
        real_step_ahead_pred_SMAPE_detail_list = [str(i) for i in real_step_ahead_pred_SMAPE_list]
        real_step_ahead_pred_SMAPE_norm_detail_list = [str(i) for i in real_step_ahead_pred_SMAPE_norm_list]
        real_step_ahead_pred_training_MSE_detail_list = [str(i) for i in real_step_ahead_pred_training_MSE_list]
        real_step_ahead_pred_SMAPE_avg_detail_list = [str(i) for i in real_step_ahead_pred_SMAPE_avg_list]
        evaluation_results['real_step_ahead_pred_SMAPE_detail'] = ','.join(real_step_ahead_pred_SMAPE_detail_list)
        evaluation_results['real_step_ahead_pred_SMAPE_norm_detail'] = ','.join(
            real_step_ahead_pred_SMAPE_norm_detail_list)
        evaluation_results['real_step_ahead_pred_training_MSE_detail'] = ','.join(
            real_step_ahead_pred_training_MSE_detail_list)
        evaluation_results['real_step_ahead_pred_SMAPE_avg_detail'] = ','.join(
            real_step_ahead_pred_SMAPE_avg_detail_list)
        evaluation_results['real_fake_step_ahead_pred_SMAPE'] = sum(real_fake_step_ahead_pred_SMAPE_list) / len(
            real_fake_step_ahead_pred_SMAPE_list)
        evaluation_results['real_fake_step_ahead_pred_SMAPE_norm'] = sum(
            real_fake_step_ahead_pred_SMAPE_norm_list) / len(real_fake_step_ahead_pred_SMAPE_norm_list)
        evaluation_results['real_fake_step_ahead_pred_SMAPE_avg'] = sum(real_fake_step_ahead_pred_SMAPE_avg_list) / len(
            real_fake_step_ahead_pred_SMAPE_avg_list)
        evaluation_results['real_fake_step_ahead_pred_training_MSE'] = sum(
            real_fake_step_ahead_pred_training_MSE_list) / len(real_fake_step_ahead_pred_training_MSE_list)
        # # store the std
        evaluation_results['real_fake_step_ahead_pred_SMAPE_std'] = np.std(real_fake_step_ahead_pred_SMAPE_list)
        evaluation_results['real_fake_step_ahead_pred_SMAPE_norm_std'] = np.std(
            real_fake_step_ahead_pred_SMAPE_norm_list)
        evaluation_results['real_fake_step_ahead_pred_training_MSE_std'] = np.std(
            real_fake_step_ahead_pred_training_MSE_list)
        evaluation_results['real_fake_step_ahead_pred_SMAPE_avg_std'] = np.std(real_fake_step_ahead_pred_SMAPE_avg_list)
        # store the detail
        real_fake_step_ahead_pred_SMAPE_detail_list = [str(i) for i in real_fake_step_ahead_pred_SMAPE_list]
        real_fake_step_ahead_pred_SMAPE_norm_detail_list = [str(i) for i in real_fake_step_ahead_pred_SMAPE_norm_list]
        real_fake_step_ahead_pred_training_MSE_detail_list = [str(i) for i in
                                                              real_fake_step_ahead_pred_training_MSE_list]
        real_fake_step_ahead_pred_SMAPE_avg_detail_list = [str(i) for i in real_fake_step_ahead_pred_SMAPE_avg_list]
        evaluation_results['real_fake_step_ahead_pred_SMAPE_detail'] = ','.join(
            real_fake_step_ahead_pred_SMAPE_detail_list)
        evaluation_results['real_fake_step_ahead_pred_SMAPE_norm_detail'] = ','.join(
            real_fake_step_ahead_pred_SMAPE_norm_detail_list)
        evaluation_results['real_fake_step_ahead_pred_training_MSE_detail'] = ','.join(
            real_fake_step_ahead_pred_training_MSE_detail_list)
        evaluation_results['real_fake_step_ahead_pred_SMAPE_avg_detail'] = ','.join(
            real_fake_step_ahead_pred_SMAPE_avg_detail_list)

        evaluation_results['discrimination_loss_test'] = sum(discrimination_loss_test_list) / len(
            discrimination_loss_test_list)
        # calcluate the std of the discrimination loss
        evaluation_results['discrimination_loss_test_std'] = np.std(discrimination_loss_test_list)
        # concate the discrimination acc to a string
        discrimination_acc_test_list = [str(i) for i in discrimination_acc_test_list]
        evaluation_results['discrimination_acc_test'] = ','.join(discrimination_acc_test_list)
        evaluation_results_high_level['Discriminator loss_test'] = evaluation_results['discrimination_loss_test']
        print('average discrimination_loss_test', evaluation_results['discrimination_loss_test'])

        # Market Facts Evaluation
        # real feature
        print(features)
        print("Running feature alignment on original data...")
        avg_ratio = feature_constraint_evaluaton(original_data_rescaled, features)
        print('Average ratio of original data that doesn\'t satisfy the logic constraints: ' + str(avg_ratio))
        # check features with logic constraints
        # features = ['open', 'high', 'low', 'close', 'adjcp', 'volume']
        # high>low and high>=open and high>=close and low<=open and low<=close
        print("Running feature alignment on generated train data...")
        avg_ratio = feature_constraint_evaluaton(generated_data_as_train_rescaled, features)
        print('Average ratio of generated train data that doesn\'t satisfy the logic constraints: ' + str(avg_ratio))
        evaluation_results['constraint_loss_train'] = avg_ratio

        print('Running feature alignment on generated test data...')
        avg_ratio = feature_constraint_evaluaton(generated_data_as_test_rescaled, features)
        print(
            f"Average of the percentage of the test data that doesn't satisfy the logic constraints: {round(avg_ratio, 4)}\n")
        evaluation_results['constraint_loss_test'] = avg_ratio
        evaluation_results_high_level['physical constraint loss'] = avg_ratio

        # Evaluation of Context Alignment

        generated_train_X_encoding = data_preprocessing_for_condition_discrimination(train_history_rescaled,
                                                                                     generated_data_as_train_rescaled,
                                                                                     train_Last_h_history,
                                                                                     args.feature,
                                                                                     differential_features=args.differential_features)
        train_X_encoding = data_preprocessing_for_condition_discrimination(train_history_rescaled,
                                                                           train_data_rescaled,
                                                                           train_Last_h_history,
                                                                           args.feature,
                                                                           differential_features=args.differential_features)
        # check diff of train_X_encoding and train_X if we have a processed train_X
        if train_X is not None:
            print('train_data_rescaled shape', train_data_rescaled.shape)
            print('train_X_encoding shape', train_X_encoding.shape)
            print('train_X preview', train_X[:1, :10, :2])
            print('train_X_encoding preview', train_X_encoding[:1, :10, :2])
            print('train_data_rescaled preview', train_data_rescaled[:1, :10, :2])
            print('train_data_rescaled preview', train_data_rescaled[:1, :10, :2])
            print('train_history_rescaled preview', train_history_rescaled[:1, 10:, :2])

            diff = train_X_encoding - train_X
            non_zero = np.nonzero(diff)
            print('number of non_zero element of X_encoding and train_X:', len(non_zero[0]))
            # print('non_zero element of diff:', test_X_encode[non_zero],test_X[non_zero])
            print("Diff sum between t X_encoding and train_X:", np.sum(diff))
        # condition prediction
        dummy_train_H = np.zeros((train_X_encoding.shape[0], 1))
        print("Running dynamic prediction on train set...")
        real_d_acc, real_l_acc, real_d_loss, real_l_loss = conditional_timegan_condition_prediction(model,
                                                                                                    X_mb=train_X_encoding,
                                                                                                    H_mb=dummy_train_H,
                                                                                                    D_mb=train_D,
                                                                                                    L_mb=train_L,
                                                                                                    args=args)
        print(
            f"Real train data dynamic prediction acc: {real_d_acc}, label prediction acc: {real_l_acc}, dynamic prediction loss: {real_d_loss}, label prediction loss: {real_l_loss}")
        fake_d_acc, fake_l_acc, fake_d_loss, fake_l_loss = conditional_timegan_condition_prediction(model,
                                                                                                    X_mb=generated_train_X_encoding,
                                                                                                    H_mb=dummy_train_H,
                                                                                                    D_mb=train_D,
                                                                                                    L_mb=train_L,
                                                                                                    args=args)
        print(
            f"Fake train data dynamic prediction acc: {fake_d_acc}, label prediction acc: {fake_l_acc}, dynamic prediction loss: {fake_d_loss}, label prediction loss: {fake_l_loss}")
        # calculate the difference between real and fake data
        d_acc_loss = real_d_acc - fake_d_acc
        l_acc_loss = real_l_acc - fake_l_acc
        d_loss_loss = fake_d_loss - real_d_loss
        l_loss_loss = fake_l_loss - real_l_loss
        # to cpu and convert to float
        d_acc_loss = d_acc_loss.item()
        l_acc_loss = l_acc_loss.item()
        d_loss_loss = d_loss_loss.item()
        l_loss_loss = l_loss_loss.item()
        print(
            f"Dynamic prediction acc loss: {d_acc_loss}, label prediction acc loss: {l_acc_loss}, dynamic prediction loss loss: {d_loss_loss}, label prediction loss loss: {l_loss_loss}")
        evaluation_results['rea_d_acc_train'] = real_d_acc
        evaluation_results['rea_l_acc_train'] = real_l_acc
        evaluation_results['rea_d_loss_train'] = real_d_loss
        evaluation_results['rea_l_loss_train'] = real_l_loss
        evaluation_results['fake_d_acc_train'] = fake_d_acc
        evaluation_results['fake_l_acc_train'] = fake_l_acc
        evaluation_results['fake_d_loss_train'] = fake_d_loss
        evaluation_results['fake_l_loss_train'] = fake_l_loss
        evaluation_results['d_acc_loss_train'] = d_acc_loss
        evaluation_results['l_acc_loss_train'] = l_acc_loss
        evaluation_results['d_loss_loss_train'] = d_loss_loss
        evaluation_results['l_loss_loss_train'] = l_loss_loss
        evaluation_results_high_level['Dynamic loss train'] = d_loss_loss
        evaluation_results_high_level['Label loss train'] = l_loss_loss
        evaluation_results_high_level['Dynamic acc loss train'] = d_acc_loss
        evaluation_results_high_level['Label acc loss train'] = l_acc_loss
        #
        print("Running dynamic prediction on test set...")

        generated_test_X_encoding = data_preprocessing_for_condition_discrimination(test_history_rescaled,
                                                                                    generated_data_as_test_rescaled,
                                                                                    test_Last_h_history, args.feature,
                                                                                    differential_features=args.differential_features)
        test_X_encoding = data_preprocessing_for_condition_discrimination(test_history_rescaled,
                                                                          test_data_rescaled,
                                                                          test_Last_h_history, args.feature,
                                                                          differential_features=args.differential_features)
        # check if test_X_encoding is the same as test_X if we have a processed test_X
        if test_X is not None:
            diff = test_X_encoding - test_X
            non_zero = np.nonzero(diff)
            print('number of non_zero element of X_encoding and test_X:', len(non_zero[0]))
            print('Sum of diff between test_X_encoding and test_X:', np.sum(diff))

        # create dummy H for the condition prediction
        dummy_test_H = np.zeros((test_X_encoding.shape[0], 1))
        real_d_acc, real_l_acc, real_d_loss, real_l_loss = conditional_timegan_condition_prediction(model,
                                                                                                    X_mb=test_X_encoding,
                                                                                                    H_mb=dummy_test_H,
                                                                                                    D_mb=test_D,
                                                                                                    L_mb=test_L,
                                                                                                    args=args)
        print(
            f"Real test data dynamic prediction acc: {real_d_acc}, label prediction acc: {real_l_acc}, dynamic prediction loss: {real_d_loss}, label prediction loss: {real_l_loss}")

        fake_d_acc, fake_l_acc, fake_d_loss, fake_l_loss = conditional_timegan_condition_prediction(model,
                                                                                                    X_mb=generated_test_X_encoding,
                                                                                                    H_mb=dummy_test_H,
                                                                                                    D_mb=test_D,
                                                                                                    L_mb=test_L,
                                                                                                    args=args)
        print(
            f"Fake test data dynamic prediction acc: {fake_d_acc}, label prediction acc: {fake_l_acc}, dynamic prediction loss: {fake_d_loss}, label prediction loss: {fake_l_loss}")
        # calculate the difference between real and fake data
        d_acc_loss = real_d_acc - fake_d_acc
        l_acc_loss = real_l_acc - fake_l_acc
        d_loss_loss = fake_d_loss - real_d_loss
        l_loss_loss = fake_l_loss - real_l_loss
        # to cpu and convert to float
        d_acc_loss = d_acc_loss.item()
        l_acc_loss = l_acc_loss.item()
        d_loss_loss = d_loss_loss.item()
        l_loss_loss = l_loss_loss.item()
        print(
            f"Dynamic prediction acc loss: {d_acc_loss}, label prediction acc loss: {l_acc_loss}, dynamic prediction loss loss: {d_loss_loss}, label prediction loss loss: {l_loss_loss}")
        evaluation_results['rea_d_acc_test'] = real_d_acc
        evaluation_results['rea_l_acc_test'] = real_l_acc
        evaluation_results['rea_d_loss_test'] = real_d_loss
        evaluation_results['rea_l_loss_test'] = real_l_loss
        evaluation_results['fake_d_acc_test'] = fake_d_acc
        evaluation_results['fake_l_acc_test'] = fake_l_acc
        evaluation_results['fake_d_loss_test'] = fake_d_loss
        evaluation_results['fake_l_loss_test'] = fake_l_loss
        evaluation_results['d_acc_loss_test'] = d_acc_loss
        evaluation_results['l_acc_loss_test'] = l_acc_loss
        evaluation_results['d_loss_loss_test'] = d_loss_loss
        evaluation_results['l_loss_loss_test'] = l_loss_loss
        evaluation_results_high_level['Dynamic loss test'] = d_loss_loss
        evaluation_results_high_level['Label loss test'] = l_loss_loss
        evaluation_results_high_level['Dynamic acc loss test'] = d_acc_loss
        evaluation_results_high_level['Label acc loss test'] = l_acc_loss

        print('All tests done!')

        # write the evaluation results to a csv where the index column is the exp_name
        evaluation_results_df = pd.DataFrame(evaluation_results, index=[args.exp])
        evaluation_results_high_level_df = pd.DataFrame(evaluation_results_high_level, index=[args.exp])
        if not result_path:
            result_path = f'{args.model_path}/{suffix}evaluation_results.csv'
            result_path_high_level = f'{args.model_path}/{suffix}evaluation_results_high_level.csv'
        else:
            # get model path from result path
            model_path = os.path.dirname(result_path)
            # get the filename of the result path without extension
            filename = os.path.splitext(os.path.basename(result_path))[0]
            # add the suffix to the filename
            filename = f'{suffix}{filename}'
            result_path = f'{model_path}/{filename}.csv'
            result_path_high_level = f'{model_path}/{filename}_high_level.csv'

        # append the results to the existing csv if it exists
        # print('result_path:',result_path,'\n')
        if os.path.exists(result_path):
            evaluation_results_df.to_csv(result_path, mode='a', header=False)
        else:
            evaluation_results_df.to_csv(result_path)
        print('result_path_high_level:', result_path_high_level, '\n')
        print('evaluation_results_high_level_df:', evaluation_results_high_level_df, '\n')
        if os.path.exists(result_path_high_level):
            evaluation_results_high_level_df.to_csv(result_path_high_level, mode='a', header=False)
        else:
            evaluation_results_high_level_df.to_csv(result_path_high_level)
        print(f"Saved evaluation results to {result_path}\n")
        print(f"Saved evaluation results to {result_path_high_level}\n")
        return result_path

    def baseline_evaluation(self, args, model, result_path, real_data_folder):
        # read, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer, train_X, test_X,
        #                              train_time, train_D, train_L, train_H, test_time, test_D, test_L, test_H
        # from dict of f'{args.model_path}/data_provider.pkl'
        print(args)
        with open(f'{args.baseline_real_data_path}/data_provider.pkl', 'rb') as f:
            data_provider_dict = pickle.load(f)
        # the keys of the dict is the variable names, the values are the variables,get the variables from the dict
        print('data_provider_dict.keys():', data_provider_dict.keys())
        dynamics_book = data_provider_dict['dynamics_book']
        dynamics_tokenizer = data_provider_dict['dynamics_tokenizer']
        label_book = data_provider_dict['label_book']
        tic_tokenizer = data_provider_dict['tic_tokenizer']
        # train_data = data_provider_dict['train_data']
        # test_data = data_provider_dict['test_data']
        train_time = data_provider_dict['train_time']
        train_D = data_provider_dict['train_D']
        train_L = data_provider_dict['train_L']
        # train_H = data_provider_dict['train_H']
        test_time = data_provider_dict['test_time']
        test_D = data_provider_dict['test_D']
        test_L = data_provider_dict['test_L']
        # test_H = data_provider_dict['test_H']
        train_Last_h = data_provider_dict['train_Last_h']
        test_Last_h = data_provider_dict['test_Last_h']
        train_scaler = data_provider_dict['train_scaler']
        test_scaler = data_provider_dict['test_scaler']
        # the datascture now is history+current, we need to split it into history and current and only use the current
        # noted that we didnot preprocess the data so the data is the original data and we dont need it here
        # train_X=train_X[:,args.history_length:args.history_length+args.max_seq_len,:]
        # test_X=test_X[:,args.history_length:args.history_length+args.max_seq_len,:]
        # train_H=train_data[:,:args.history_length,:]
        # test_H=test_data[:,:args.history_length,:]

        train_D_word = dynamics_tokenizer.one_hot_to_word(train_D)
        train_L_word = tic_tokenizer.one_hot_to_word(train_L)
        test_D_word = dynamics_tokenizer.one_hot_to_word(test_D)
        test_L_word = tic_tokenizer.one_hot_to_word(test_L)
        all_result_csv_paths = set()

        # benchmarking baselines from the result_paths
        # The file structure is as follows:
        # Dataset name: DJ30
        # Tic+Dynmaic:  GOOG_0
        # Experiment seed: seed=0
        # Model Name: SigCWGAN
        # parse the folder name to get the dataset name, model name, and experiment seed
        # get the dataset name
        dataset_name = os.path.basename(os.path.dirname(result_path))
        # print('dataset_name:',dataset_name)
        # get all the sub folder paths
        Experiment_paths = [f.path for f in os.scandir(result_path) if f.is_dir()]
        # print('Experiment_paths:',Experiment_paths)
        for experiment_path in Experiment_paths:
            # get the subfolder name
            # experiment_name = os.path.basename(os.path.dirname(experiment_path))
            experiment_name = os.path.basename(experiment_path)
            # parse the tic and dynamic from experiment_name with '_'
            print(experiment_name)
            tic, dynamic = experiment_name.split('_')
            # real all the sub folder name and paths under experiment_path
            experiment_seeds_paths = [f.path for f in os.scandir(experiment_path) if f.is_dir()]
            # print('experiment_seeds_paths:',experiment_seeds_paths)
            result_folder = f'{args.baseline_real_data_path}/evaluation_result'
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            for experiment_seeds_path in experiment_seeds_paths:

                # read the generated data of each model
                # get all model names and paths
                model_paths = [f.path for f in os.scandir(experiment_seeds_path) if f.is_dir()]
                # print('model_paths:',model_paths)
                for model_path in model_paths:
                    # model_name=os.path.basename(os.path.dirname(model_path))
                    model_name = os.path.basename(model_path)
                    # read the generated data
                    generated_train_path = f'{model_path}/{model_name}_{tic}_{dynamic}_generated_train.npy'
                    generated_test_path = f'{model_path}/{model_name}_{tic}_{dynamic}_generated_test.npy'
                    # find the index of the data where train_D==dynamic and train_L==tic from train_D_word and train_L_word(lists)
                    train_index = []
                    for i in range(len(train_D_word)):
                        if train_D_word[i] == str(dynamic) and train_L_word[i] == str(tic):
                            train_index.append(i)
                    train_index = np.array(train_index)
                    # find the index of the data where train_D==dynamic and train_L==tic from test_D_word and test_L_word(lists)
                    test_index = []
                    for i in range(len(test_D_word)):
                        if test_D_word[i] == str(dynamic) and test_L_word[i] == str(tic):
                            test_index.append(i)
                    test_index = np.array(test_index)
                    if args.prepare_encoded_data:
                        generated_train_X = np.load(generated_train_path)
                        generated_test_X = np.load(generated_test_path)
                        print('shape of generated_train_')
                        print('generated_train_X.shape:', generated_train_X.shape)
                        print('generated_test_X.shape:', generated_test_X.shape)
                        print('train_scaler.shape:', train_scaler.shape, train_Last_h.shape)
                        train_scaler_slice = train_scaler[train_index]
                        test_scaler_slice = test_scaler[test_index]
                        train_Last_h_slice = train_Last_h[train_index]
                        test_Last_h_slice = test_Last_h[test_index]
                        train_Last_h_data_slice, train_Last_h_history_slice = np.split(train_Last_h_slice, 2, axis=1)
                        test_Last_h_data_slice, test_Last_h_history_slice = np.split(test_Last_h_slice, 2, axis=1)
                        generated_data_as_train_rescaled = conditional_rescale_data(generated_train_X,
                                                                                    train_scaler_slice,
                                                                                    args.differential_features,
                                                                                    train_Last_h_data_slice,
                                                                                    args.scaler_order,
                                                                                    original_feature_order=args.feature)
                        generated_data_as_test_rescaled = conditional_rescale_data(generated_test_X, test_scaler_slice,
                                                                                   args.differential_features,
                                                                                   test_Last_h_data_slice,
                                                                                   args.scaler_order,
                                                                                   original_feature_order=args.feature)
                    else:
                        generated_data_as_train_rescaled = np.load(generated_train_path)
                        generated_data_as_test_rescaled = np.load(generated_test_path)
                    result_csv_path = f'{result_folder}/{model_name}_evaluation_results.csv'
                    # append to set all_result_csv_paths
                    all_result_csv_paths.add(result_csv_path)
                    # create a empty result csv if it does not exist
                    # if not os.path.exists(result_csv_path):
                    # # create an empty dataframe
                    #     evaluation_results_df = pd.DataFrame()
                    #     evaluation_results_df.to_csv(result_csv_path)

                    # read the true data from the real_data_folder
                    real_data_train_path = f'{real_data_folder}/{tic}_{dynamic}_train.npy'
                    real_data_test_path = f'{real_data_folder}/{tic}_{dynamic}_test.npy'
                    if args.prepare_encoded_data:
                        train_data_with_history = np.load(real_data_train_path)
                        test_data_with_history = np.load(real_data_test_path)
                        # get the train_history_rescaled and test_history_rescaled
                        train_history = train_data_with_history[:, :args.history_length, :]
                        test_history = test_data_with_history[:, :args.history_length, :]
                        # get the train_data_rescaled and test_data_rescaled
                        train_data = train_data_with_history[:, args.history_length:, :]
                        test_data = test_data_with_history[:, args.history_length:, :]

                        train_data_rescaled = conditional_rescale_data(train_data, train_scaler_slice,
                                                                       args.differential_features,
                                                                       train_Last_h_data_slice, args.scaler_order,
                                                                       original_feature_order=args.feature)
                        test_data_rescaled = conditional_rescale_data(test_data, test_scaler_slice,
                                                                      args.differential_features,
                                                                      test_Last_h_data_slice, args.scaler_order,
                                                                      original_feature_order=args.feature)
                        train_history_rescaled = conditional_rescale_data(train_history, train_scaler_slice,
                                                                          args.differential_features,
                                                                          train_Last_h_history_slice, args.scaler_order,
                                                                          original_feature_order=args.feature)
                        test_history_rescaled = conditional_rescale_data(test_history, test_scaler_slice,
                                                                         args.differential_features,
                                                                         test_Last_h_history_slice, args.scaler_order,
                                                                         original_feature_order=args.feature)


                    else:
                        train_data_rescaled_with_history = np.load(real_data_train_path)
                        test_data_rescaled_with_history = np.load(real_data_test_path)

                        # get the train_history_rescaled and test_history_rescaled
                        train_history_rescaled = train_data_rescaled_with_history[:, :args.history_length, :]
                        test_history_rescaled = test_data_rescaled_with_history[:, :args.history_length, :]
                        # get the train_data_rescaled and test_data_rescaled
                        train_data_rescaled = train_data_rescaled_with_history[:, args.history_length:, :]
                        test_data_rescaled = test_data_rescaled_with_history[:, args.history_length:, :]

                    train_time_slice = list(np.array(train_time)[train_index])
                    test_time_slice = list(np.array(test_time)[test_index])
                    train_D_slice = train_D[train_index]
                    test_D_slice = test_D[test_index]
                    train_L_slice = train_L[train_index]
                    test_L_slice = test_L[test_index]
                    train_Last_h_slice = train_Last_h[train_index]
                    test_Last_h_slice = test_Last_h[test_index]
                    print(f'evaluating {dataset_name}, {tic}, {dynamic}, {model_name}, seed={args.exp}')
                    # check if the size of train_data_rescaled is equal to generated_data_as_train_rescaled
                    assert train_data_rescaled.shape[0] == generated_data_as_train_rescaled.shape[0]
                    _ = self.evaluate(args, model, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer,
                                      None, None,
                                      None, None,
                                      generated_data_as_train_rescaled, generated_data_as_test_rescaled,
                                      train_data_rescaled, test_data_rescaled,
                                      train_history_rescaled, test_history_rescaled,
                                      train_Last_h_slice, test_Last_h_slice,
                                      train_time_slice, train_D_slice, train_L_slice, None,
                                      test_time_slice, test_D_slice, test_L_slice, None,
                                      args.exp, result_path=result_csv_path, model_name=model_name)
        # summerize the results of each model
        for result_path in all_result_csv_paths:
            print(f'summerizing the results of {result_path})')
            result_summary(result_path)
        # summerize the results of all models into a single csv
        print(f'summerizing the results of all models into a single csv')
        baseline_results_path = f'{args.baseline_real_data_path}/baseline_evaluation_results_high_level.csv'
        for result_path in all_result_csv_paths:
            # parse the model name from the result_path
            model_name = os.path.basename(result_path).split('_')[0]
            model_path = os.path.dirname(result_path)
            # get the filename without extension name
            file_name = os.path.basename(result_path).split('.')[0]
            result_path_high_level = f'{model_path}/{file_name}_high_level.csv'
            # read the result_path_high_level into a dataframe with index
            df = pd.read_csv(result_path_high_level, index_col=0)
            # get the row with 'mean' as index
            mean_row = df.loc['mean']
            # if baseline_results_path does not exist, create it
            if not os.path.exists(baseline_results_path):
                # give the same column names as the df
                df_baseline = pd.DataFrame(columns=df.columns)
                # save the mean_row to the df_baseline with model_name as index
                df_baseline.loc[model_name] = mean_row
                # save the df_baseline to the baseline_results_path
                df_baseline.to_csv(baseline_results_path)
            else:
                # append the mean_row to the baseline_results_path with index as model_name
                df_baseline = pd.read_csv(baseline_results_path, index_col=0)
                df_baseline.loc[model_name] = mean_row
                # save the df_baseline to the baseline_results_path
                df_baseline.to_csv(baseline_results_path)
        print('summary of all models is saved to ', baseline_results_path)

    def aggregation_baseline_evaluation(self, args, model, result_path, real_data_folder):
        # read, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer, train_X, test_X,
        #                              train_time, train_D, train_L, train_H, test_time, test_D, test_L, test_H
        # from dict of f'{args.model_path}/data_provider.pkl'
        with open(f'{args.baseline_real_data_path}/data_provider.pkl', 'rb') as f:
            data_provider_dict = pickle.load(f)
        # the keys of the dict is the variable names, the values are the variables,get the variables from the dict
        dynamics_book = data_provider_dict['dynamics_book']
        dynamics_tokenizer = data_provider_dict['dynamics_tokenizer']
        label_book = data_provider_dict['label_book']
        tic_tokenizer = data_provider_dict['tic_tokenizer']
        train_time = data_provider_dict['train_time']
        train_D = data_provider_dict['train_D']
        train_L = data_provider_dict['train_L']
        # train_H = data_provider_dict['train_H']
        test_time = data_provider_dict['test_time']
        test_D = data_provider_dict['test_D']
        test_L = data_provider_dict['test_L']
        # test_H = data_provider_dict['test_H']
        train_Last_h = data_provider_dict['train_Last_h']
        test_Last_h = data_provider_dict['test_Last_h']
        train_scaler = data_provider_dict['train_scaler']
        test_scaler = data_provider_dict['test_scaler']

        train_D_word = dynamics_tokenizer.one_hot_to_word(train_D)
        train_L_word = tic_tokenizer.one_hot_to_word(train_L)
        test_D_word = dynamics_tokenizer.one_hot_to_word(test_D)
        test_L_word = tic_tokenizer.one_hot_to_word(test_L)
        all_result_csv_paths = set()

        dataset_name = os.path.basename(os.path.dirname(result_path))
        # print('dataset_name:',dataset_name)
        # get all the sub folder paths
        Experiment_paths = [f.path for f in os.scandir(result_path) if f.is_dir()]

        # get the tic and dynamic list from the Experiment_paths
        tic_list = []
        dynamic_list = []
        model_list = ['GMMN', 'SigCWGAN', 'TimeGAN', 'RCGAN']
        print('model_list', model_list)
        for experiment_path in Experiment_paths:
            # get the subfolder name
            # experiment_name = os.path.basename(os.path.dirname(experiment_path))
            experiment_name = os.path.basename(experiment_path)
            # parse the tic and dynamic from experiment_name with '_'
            print(experiment_name)
            tic, dynamic = experiment_name.split('_')
            tic_list.append(tic)
            dynamic_list.append(dynamic)
        # print('Experiment_paths:',Experiment_paths)
        for model_name_to_eval in model_list:
            # create a placeholder for the generated_data_as_train_rescaled, generated_data_as_test_rescaled,
            #                                                   train_data_rescaled, test_data_rescaled,
            #                                                   train_history_rescaled,test_history_rescaled,
            # which has the same samples of train_D but different size of features
            result_folder = f'{args.baseline_real_data_path}/aggregation_evaluation_result'
            result_csv_path = f'{result_folder}/{model_name_to_eval}_evaluation_results.csv'
            all_result_csv_paths.add(result_csv_path)
            # create a placeholder for the generated_data_as_train_rescaled, generated_data_as_test_rescaled, where sample size is the same as train_D and last two dimensions are  args.max_seq_len(args.history_length) ,args.feature_dim
            generated_data_as_train_rescaled = np.zeros((train_D.shape[0], args.max_seq_len, args.feature_dim))
            generated_data_as_test_rescaled = np.zeros((test_D.shape[0], args.max_seq_len, args.feature_dim))
            # create a placeholder for the train_data_rescaled, test_data_rescaled, where sample size is the same as train_D and last two dimensions are  args.max_seq_len(args.history_length) ,args.feature_dim
            train_data_rescaled = np.zeros((train_D.shape[0], args.max_seq_len, args.feature_dim))
            test_data_rescaled = np.zeros((test_D.shape[0], args.max_seq_len, args.feature_dim))
            # create a placeholder for the train_history_rescaled, test_history_rescaled, where sample size is the same as train_D and last two dimensions are  args.history_length ,args.feature_dim
            train_history_rescaled = np.zeros((train_D.shape[0], args.history_length, args.feature_dim))
            test_history_rescaled = np.zeros((test_D.shape[0], args.history_length, args.feature_dim))
            # print('generated_data_as_train_rescaled',generated_data_as_train_rescaled.shape)
            load_size = 0
            for experiment_path in Experiment_paths:
                # get the subfolder name
                # experiment_name = os.path.basename(os.path.dirname(experiment_path))
                experiment_name = os.path.basename(experiment_path)
                # parse the tic and dynamic from experiment_name with '_'
                print(experiment_name)
                tic, dynamic = experiment_name.split('_')
                # find the index of the data where train_D==dynamic and train_L==tic from train_D_word and train_L_word(lists)
                train_index = []
                for i in range(len(train_D_word)):
                    if train_D_word[i] == str(dynamic) and train_L_word[i] == str(tic):
                        train_index.append(i)
                train_index = np.array(train_index)
                # find the index of the data where train_D==dynamic and train_L==tic from test_D_word and test_L_word(lists)
                test_index = []
                for i in range(len(test_D_word)):
                    if test_D_word[i] == str(dynamic) and test_L_word[i] == str(tic):
                        test_index.append(i)
                test_index = np.array(test_index)
                # real all the sub folder name and paths under experiment_path
                experiment_seeds_paths = [f.path for f in os.scandir(experiment_path) if f.is_dir()]
                # print('experiment_seeds_paths:',experiment_seeds_paths)
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                for experiment_seeds_path in experiment_seeds_paths:

                    # read the generated data of each model
                    # get all model names and paths
                    model_paths = [f.path for f in os.scandir(experiment_seeds_path) if f.is_dir()]
                    for model_path in model_paths:
                        model_name = os.path.basename(model_path)
                        if model_name != model_name_to_eval:
                            continue
                        print(f'load {model_name}')
                        # read the generated data
                        generated_train_path = f'{model_path}/{model_name}_{tic}_{dynamic}_generated_train.npy'
                        generated_test_path = f'{model_path}/{model_name}_{tic}_{dynamic}_generated_test.npy'
                        print('generated_train_path', generated_train_path)

                        if args.prepare_encoded_data:
                            generated_train_X = np.load(generated_train_path)
                            generated_test_X = np.load(generated_test_path)
                            train_scaler_slice = train_scaler[train_index]
                            test_scaler_slice = test_scaler[test_index]
                            train_Last_h_slice = train_Last_h[train_index]
                            test_Last_h_slice = test_Last_h[test_index]
                            train_Last_h_data_slice, train_Last_h_history_slice = np.split(train_Last_h_slice, 2,
                                                                                           axis=1)
                            test_Last_h_data_slice, test_Last_h_history_slice = np.split(test_Last_h_slice, 2, axis=1)
                            generated_data_as_train_rescaled_slice = conditional_rescale_data(generated_train_X,
                                                                                              train_scaler_slice,
                                                                                              args.differential_features,
                                                                                              train_Last_h_data_slice,
                                                                                              args.scaler_order,
                                                                                              original_feature_order=args.feature)
                            generated_data_as_test_rescaled_slice = conditional_rescale_data(generated_test_X,
                                                                                             test_scaler_slice,
                                                                                             args.differential_features,
                                                                                             test_Last_h_data_slice,
                                                                                             args.scaler_order,
                                                                                             original_feature_order=args.feature)
                        else:
                            generated_data_as_train_rescaled_slice = np.load(generated_train_path)
                            generated_data_as_test_rescaled_slice = np.load(generated_test_path)

                        # read the true data from the real_data_folder
                        real_data_train_path = f'{real_data_folder}/{tic}_{dynamic}_train.npy'
                        real_data_test_path = f'{real_data_folder}/{tic}_{dynamic}_test.npy'

                        if args.prepare_encoded_data:
                            train_data_with_history = np.load(real_data_train_path)
                            test_data_with_history = np.load(real_data_test_path)
                            # get the train_history_rescaled and test_history_rescaled
                            train_history = train_data_with_history[:, :args.history_length, :]
                            test_history = test_data_with_history[:, :args.history_length, :]
                            # get the train_data_rescaled and test_data_rescaled
                            train_data = train_data_with_history[:, args.history_length:, :]
                            test_data = test_data_with_history[:, args.history_length:, :]

                            train_data_rescaled_slice = conditional_rescale_data(train_data, train_scaler_slice,
                                                                                 args.differential_features,
                                                                                 train_Last_h_data_slice,
                                                                                 args.scaler_order,
                                                                                 original_feature_order=args.feature)
                            test_data_rescaled_slice = conditional_rescale_data(test_data, test_scaler_slice,
                                                                                args.differential_features,
                                                                                test_Last_h_data_slice,
                                                                                args.scaler_order,
                                                                                original_feature_order=args.feature)
                            train_history_rescaled_slice = conditional_rescale_data(train_history, train_scaler_slice,
                                                                                    args.differential_features,
                                                                                    train_Last_h_history_slice,
                                                                                    args.scaler_order,
                                                                                    original_feature_order=args.feature)
                            test_history_rescaled_slice = conditional_rescale_data(test_history, test_scaler_slice,
                                                                                   args.differential_features,
                                                                                   test_Last_h_history_slice,
                                                                                   args.scaler_order,
                                                                                   original_feature_order=args.feature)
                        else:
                            train_data_rescaled_with_history_slice = np.load(real_data_train_path)
                            test_data_rescaled_with_history_slice = np.load(real_data_test_path)

                            # get the train_history_rescaled and test_history_rescaled
                            train_history_rescaled_slice = train_data_rescaled_with_history_slice[:,
                                                           :args.history_length, :]
                            test_history_rescaled_slice = test_data_rescaled_with_history_slice[:, :args.history_length,
                                                          :]
                            # get the train_data_rescaled and test_data_rescaled
                            train_data_rescaled_slice = train_data_rescaled_with_history_slice[:, args.history_length:,
                                                        :]
                            test_data_rescaled_slice = test_data_rescaled_with_history_slice[:, args.history_length:, :]

                        load_size += len(train_index)
                        # assign the slice of data to the corresponding index
                        generated_data_as_train_rescaled[train_index] = generated_data_as_train_rescaled_slice
                        generated_data_as_test_rescaled[test_index] = generated_data_as_test_rescaled_slice
                        train_data_rescaled[train_index] = train_data_rescaled_slice
                        test_data_rescaled[test_index] = test_data_rescaled_slice
                        train_history_rescaled[train_index] = train_history_rescaled_slice
                        test_history_rescaled[test_index] = test_history_rescaled_slice
                        print(f'load {load_size} sample for {model_name_to_eval}')

            print(f'evaluating {dataset_name} {model_name_to_eval}, seed={args.exp}')
            print('generated_data_as_train_rescaled:', generated_data_as_train_rescaled.shape)
            print('generated_data_as_test_rescaled:', generated_data_as_test_rescaled.shape)
            print('train_data_rescaled:', train_data_rescaled.shape)
            print('test_data_rescaled:', test_data_rescaled.shape)
            print('train_history_rescaled:', train_history_rescaled.shape)
            print('test_history_rescaled:', test_history_rescaled.shape)
            print('train_Last_h:', train_Last_h.shape)
            print('test_Last_h:', test_Last_h.shape)
            print('train_time:', train_time.shape)
            print('test_time:', test_time.shape)
            print('train_D:', train_D.shape)
            print('test_D:', test_D.shape)
            print('train_L:', train_L.shape)
            print('test_L:', test_L.shape)
            if args.prepare_encoded_data:
                suffix = 'encoded_agg_'
            else:
                suffix = 'agg_'
            _ = self.evaluate(args, model, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer,
                              None, None,
                              None, None,
                              generated_data_as_train_rescaled, generated_data_as_test_rescaled,
                              train_data_rescaled, test_data_rescaled,
                              train_history_rescaled, test_history_rescaled,
                              train_Last_h, test_Last_h,
                              train_time, train_D, train_L, None,
                              test_time, test_D, test_L, None,
                              args.exp, result_path=result_csv_path, model_name=suffix + model_name_to_eval)
