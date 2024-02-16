## Necessary packages

import warnings

import numpy as np

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
import os
from data.utils import *


def data_sample_worker(data, seq_len, features, history_length, full_data):
    # get the data sample by dynamic and tic and save them to npy file
    intervals = get_intervals(data)
    data_sample_list = []
    historical_data_list = []
    # get the starting index of each interval
    start_index_by_interval = []
    index = data['index']
    last_hist_vec = []
    for interval in intervals:
        start_index_by_interval.append(index[interval[0]])

    for interval_index, interval in enumerate(intervals):
        data_seg = data.iloc[interval[0]:interval[1], :]
        data_seg = data_seg.loc[:, features]
        if len(data_seg) - seq_len <= 0:
            continue
        for i in range(0, len(data_seg) - seq_len):

            # get the history data
            if interval[0] + i - history_length < 0:
                continue
            # get the history sample
            history_start = start_index_by_interval[interval_index] + i - history_length
            history_end = start_index_by_interval[interval_index] + i
            history_sample = full_data.iloc[history_start:history_end, :]
            history_sample = history_sample.loc[:, features]

            # get last low before data sample
            try:
                last_low_befor_data_sample = data.iloc[interval[0] + i - 1, :]['low']
                # do the same thing to history_sample
                last_low_befor_history_sample = full_data.iloc[history_start - 1, :]['low']
                last_hist_vec.append(np.array([last_low_befor_data_sample, last_low_befor_history_sample]))
            except:
                continue

            # get the data sample
            sample_start = start_index_by_interval[interval_index] + i - history_length
            sample_end = start_index_by_interval[interval_index] + i + seq_len
            data_sample = full_data.iloc[sample_start:sample_end, :]
            data_sample = data_sample.loc[:, features]
            data_sample_list.append(data_sample)
            historical_data_list.append(history_sample)
    return data_sample_list, historical_data_list, last_hist_vec


def get_tics(path):
    data = pd.read_csv(path).reset_index()
    tics = data['tic'].unique()
    return tics


def data_provider(data_file_path, tics, args, tic_tokenizer, dynamics_tokenizer, features, history_length):
    # 1.get the data by stock/dynamics type
    # 2. get data sample (and history) of the same length
    # 3. StandardScaler the data by itself

    data = pd.read_csv(data_file_path).reset_index()
    # process data column to lower case
    data.columns = map(str.lower, data.columns)

    regime_num = len(data['label'].unique())

    # list for X, T, D, L,H scaler vector

    # get the file name of the data_file_path
    file_name = data_file_path.split('/')[-1]
    # get the folder path of the data_file_path
    folder_path = data_file_path.split(file_name)[0]
    # create a folder under the same folder of data_file_path with the file_name without extension
    seg_folder_name = file_name.split('.')[0]
    seg_folder_path = folder_path + seg_folder_name
    if not os.path.exists(seg_folder_path):
        os.makedirs(seg_folder_path)
    # clean segs folder
    for file in os.listdir(seg_folder_path):
        os.remove(seg_folder_path + '/' + file)
    X_list = []
    D_list = []
    L_list = []
    H_list = []
    Last_h_list = []
    print('full data shape: ', data.shape)
    for tic in tqdm(tics):
        print(tic)
        for j in range(regime_num):
            # data_sample_list_vec = []
            data_seg = data.loc[(data['tic'] == tic) & (data['label'] == j), ['index'] + features]
            data_seg.to_csv(f'{seg_folder_path}/data_seg_{tic}_{j}.csv')
            data_seg = pd.read_csv(f'{seg_folder_path}/data_seg_{tic}_{j}.csv')

            data_samples, historcial_samples, last_hist_vec = data_sample_worker(data_seg, args.max_seq_len, features,
                                                                                 history_length,
                                                                                 data)  # list of dataframes dim: (n,seq_len,features) time: (n,1)
            tic_token = tic_tokenizer.word_to_one_hot(tic)
            # scaler_vector = [min_sclar_by_tic[tic], max_sclar_by_tic[tic]]
            tic_token = np.array(tic_token, dtype=float)
            tic_token = np.expand_dims(tic_token, axis=0)
            tic_token = np.repeat(tic_token, len(data_samples), axis=0)  # dim: (n,tic_token)
            dynmamic_token = dynamics_tokenizer.word_to_one_hot(str(j))
            dynmamic_token = np.array(dynmamic_token, dtype=float)
            dynmamic_token = np.expand_dims(dynmamic_token, axis=0)
            dynmamic_token = np.repeat(dynmamic_token, len(data_samples), axis=0)  # dim:
            # broadcast tic and dynamics token and scaler to each data slice
            data_samples = np.array(data_samples)
            historcial_samples = np.array(historcial_samples)
            print('sample number of dynamic {} and tic {} is {}'.format(j, tic, len(data_samples)))
            X_list.append(data_samples)
            D_list.append(dynmamic_token)
            L_list.append(tic_token)
            H_list.append(historcial_samples)
            Last_h_list.append(last_hist_vec)

    X = np.concatenate(X_list, axis=0)  # dim: (n,seq_len,features)
    D = np.vstack(D_list)  # dim: (n,dynmamic_token)
    L = np.vstack(L_list)  # dim: (n,tic_token)
    H = np.concatenate(H_list, axis=0)  # dim: (n,historical_len,features)
    Last_H = np.vstack(Last_h_list)  # dim: (n,2)
    return X, D, L, H, Last_H
