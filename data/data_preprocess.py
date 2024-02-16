## Necessary packages

import warnings

import numpy as np

warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from data.utils import *


def data_sample_worker(data, seq_len, features, history_length, full_data, differential_features=True):
    time = []
    intervals = get_intervals(data)

    normalized_data_samples = []
    normalized_history_samples = []
    original_data_samples = []
    original_history_samples = []
    # get the starting index of each interval
    start_index_by_interval = []
    last_hist_vec = []
    scaler_vec = []
    index = data['index']
    for interval in intervals:
        start_index_by_interval.append(index[interval[0]])

    for interval_index, interval in enumerate(intervals):
        data_seg = data.iloc[interval[0]:interval[1], :]
        data_seg = data_seg.loc[:, features]
        # normalization
        scaler = MinMaxScaler()
        scaler.fit(data_seg)
        if len(data_seg) - seq_len <= 0:
            continue
        for i in range(0, len(data_seg) - seq_len):
            data_sample = data_seg.iloc[i:i + seq_len, :]

            # get the history data
            if interval[0] + i - history_length < 0:
                continue
            history_start = start_index_by_interval[interval_index] + i - history_length
            history_end = start_index_by_interval[interval_index] + i
            history_sample = full_data.iloc[history_start:history_end, :]
            history_sample = history_sample.loc[:, features]

            try:
                last_low_befor_data_sample = data.iloc[interval[0] + i - 1, :]['low']
                # do the same thing to history_sample
                last_low_befor_history_sample = full_data.iloc[history_start - 1, :]['low']
                last_hist_vec.append(np.array([last_low_befor_data_sample, last_low_befor_history_sample]))
            except:
                continue

            check_data_processing = True
            if check_data_processing == True:
                ori_data_sample = data_sample.copy()
                ori_hitory_sample = history_sample.copy()
            else:
                ori_data_sample = None
                ori_hitory_sample = None
            # normalization
            normalized_data_sample = scaler.transform(data_sample)
            normalized_history_sample = scaler.transform(history_sample)
            time.append(seq_len)
            normalized_data_samples.append(normalized_data_sample)
            normalized_history_samples.append(normalized_history_sample)
            original_data_samples.append(ori_data_sample)
            original_history_samples.append(ori_hitory_sample)
            scaler_vec.append(scaler)
    return normalized_data_samples, normalized_history_samples, time, scaler_vec, last_hist_vec, original_data_samples, original_history_samples


def dataset_preprocessing(data_file_path, tics, args, tic_tokenizer, dynamics_tokenizer, features, history_length):
    # 1.get the data by stock/dynamics type
    # 2. get data sample (and history) of the same length
    # 3. StandardScaler the data by itself

    data = pd.read_csv(data_file_path).reset_index()
    # process data column to lower case
    data.columns = map(str.lower, data.columns)

    regime_num = len(data['label'].unique())

    # list for X, T, D, L,H scaler vector
    X_list = []
    T_list = []
    D_list = []
    L_list = []
    H_list = []
    OX_list = []
    OH_list = []
    Last_h_list = []

    scaler_vec_list = []

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

    print('full data shape: ', data.shape)
    print('Using differential features: ', args.differential_features)
    for tic in tqdm(tics):
        print(tic)
        for j in range(regime_num):
            data_seg = data.loc[(data['tic'] == tic) & (data['label'] == j), ['index'] + features]
            data_seg.to_csv(f'{seg_folder_path}/data_seg_{tic}_{j}.csv')
            data_seg = pd.read_csv(f'{seg_folder_path}/data_seg_{tic}_{j}.csv')
            data_samples, historcial_samples, time, scaler_vec, last_hist_vec, original_data_samples, original_history_samples = data_sample_worker(
                data_seg, args.max_seq_len, features, history_length, data,
                args.differential_features)  # list of dataframes dim: (n,seq_len,features) time: (n,1)
            # broadcast tic and dynamics token and scaler to each data slice
            tic_token = tic_tokenizer.word_to_one_hot(tic)
            tic_token = np.array(tic_token, dtype=float)
            tic_token = np.expand_dims(tic_token, axis=0)
            tic_token = np.repeat(tic_token, len(data_samples), axis=0)  # dim: (n,tic_token)
            dynmamic_token = dynamics_tokenizer.word_to_one_hot(str(j))
            dynmamic_token = np.array(dynmamic_token, dtype=float)
            dynmamic_token = np.expand_dims(dynmamic_token, axis=0)
            dynmamic_token = np.repeat(dynmamic_token, len(data_samples), axis=0)  # dim: (n,dynmamic_token)
            data_samples = np.array(data_samples)
            historcial_samples = np.array(historcial_samples)
            print('sample number of dynamic {} and tic {} is {}'.format(j, tic, len(data_samples)))
            X_list.append(data_samples)
            T_list.extend(time)
            D_list.append(dynmamic_token)
            L_list.append(tic_token)
            H_list.append(historcial_samples)
            Last_h_list.append(last_hist_vec)
            OX_list.append(original_data_samples)
            OH_list.append(original_history_samples)
            scaler_vec_list.extend(scaler_vec)

    # stack all data slices
    # concatenate X_list
    X = np.concatenate(X_list, axis=0)  # dim: (n,seq_len,features)
    H = np.concatenate(H_list, axis=0)  # dim: (n,historical_len,features)
    T = T_list  # dim: (n)
    D = np.vstack(D_list)  # dim: (n,dynmamic_token)
    L = np.vstack(L_list)  # dim: (n,tic_token)
    OX = np.concatenate(OX_list, axis=0)  # dim: (n,seq_len,features)
    OH = np.concatenate(OH_list, axis=0)  # dim: (n,historical_len,features)
    Last_h = np.vstack(Last_h_list)  # dim: (n,2)
    # shape and dtype of X
    print("X.shape: ", X.shape, " X.dtype: ", X.dtype)
    print('H.shape: ', H.shape, ' H.dtype: ', H.dtype)
    print("T.shape: ", np.array(T).shape, " T.dtype: ", np.array(T).dtype)
    print("D.shape: ", D.shape, " D.dtype: ", D.dtype)
    print("L.shape: ", L.shape, " L.dtype: ", L.dtype)
    print("Last_h.shape: ", Last_h.shape, " Last_h.dtype: ", Last_h.dtype)
    # stack the scalr list to a scaler vector for each sample
    scaler = np.array(scaler_vec_list)
    print("scaler.shape: ", scaler.shape, " scaler.dtype: ", scaler.dtype)
    # save the order of the scaler
    scaler_order = ['low', 'O_minus_L', 'C_minus_L', 'H_minus_maxOC']

    return X, H, T, D, L, scaler, scaler_order, Last_h, OX, OH


def rescale_data(ori_data, scaler):
    # recale the data to the original scale
    # ori_data: (n,seq_len,features)
    # scaler: (n)
    data = []
    for i in range(len(ori_data)):
        data.append(scaler[i].inverse_transform(ori_data[i]))
    # to numpy array
    data = np.array(data)
    return data
