# -*- coding: UTF-8 -*-
# Local modules
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

# 3rd-Party Modules
import torch
import pandas as pd

from utils.util import *
from layers.Tokenizer import Tokenizer
from downstream_tasks.downstream_utils.data_preprocessing import *


def worker(data_path, args):
    #########################
    # Load and preprocess data for model
    #########################

    tics = get_tics(data_path)
    args.label_dim = len(tics)

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

    data = pd.read_csv(data_path).reset_index()
    # process data column to lower case
    data.columns = map(str.lower, data.columns)

    dynamics_num = len(data['label'].unique())

    # get the file name of the data_file_path
    file_name = data_path.split('/')[-1]
    # get the folder path of the data_file_path
    folder_path = data_path.split(file_name)[0]
    # create a folder under the same folder of data_file_path with the file_name without extension
    seg_folder_name = file_name.split('.')[0]
    seg_folder_path = folder_path + seg_folder_name
    if not os.path.exists(seg_folder_path):
        os.makedirs(seg_folder_path)
    # clean segs folder
    for file in os.listdir(seg_folder_path):
        os.remove(seg_folder_path + '/' + file)
    print('full data shape: ', data.shape)

    # create downstream_tasks/data/original and downstream_tasks/data/processed folder if not exist
    if not os.path.exists('downstream_tasks/data/original'):
        os.makedirs('downstream_tasks/data/original')
    if not os.path.exists('downstream_tasks/data/processed'):
        os.makedirs('downstream_tasks/data/processed')

    for tic in tqdm(tics):
        print(tic)
        for j in range(dynamics_num):
            data_seg = data.loc[(data['tic'] == tic) & (data['label'] == j), ['index', 'date'] + features]
            data_seg.to_csv(f'{seg_folder_path}/data_seg_{tic}_{j}.csv')
            data_seg = pd.read_csv(f'{seg_folder_path}/data_seg_{tic}_{j}.csv')
            # get the train data and test data split by tic and dynamic
            original_data_sample_train, original_data_sample_test, normalized_data_sample_train, normalized_data_sample_test = data_sample_worker(
                data_seg, args.max_seq_len, features, args.history_length, data, tic, j,
                args)  # list of dataframes dim: (n,seq_len,features) time: (n,1)

            # save the samples into pickle file
            with open(f'downstream_tasks/data/original/data_sample_train_{tic}_{j}.pkl', 'wb') as f:
                pickle.dump(original_data_sample_train, f)
            with open(f'downstream_tasks/data/original/data_sample_test_{tic}_{j}.pkl', 'wb') as f:
                pickle.dump(original_data_sample_test, f)
            with open(f'downstream_tasks/data/processed/data_sample_train_{tic}_{j}.pkl', 'wb') as f:
                pickle.dump(normalized_data_sample_train, f)
            with open(f'downstream_tasks/data/processed/data_sample_test_{tic}_{j}.pkl', 'wb') as f:
                pickle.dump(normalized_data_sample_test, f)


if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        default='cuda',
        type=str)
    parser.add_argument(
        '--exp',
        default='test',
        type=str)
    parser.add_argument(
        '--seed',
        default=-1,
        type=int)
    # Data Arguments
    parser.add_argument(
        '--max_seq_len',
        default=100,
        type=int)
    parser.add_argument(
        '--train_rate',
        default=0.5,
        type=float)
    parser.add_argument(
        '--data_path',
        default='',
        type=str)
    parser.add_argument(
        '--dynamic_dim',
        default=3,
        type=int)
    parser.add_argument(
        '--label_dim',
        default=29,
        type=int)
    parser.add_argument(
        '--history_length',
        type=int,
        default=30
    )
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    worker(args.data_path, args)
