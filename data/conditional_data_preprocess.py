## Necessary packages

import warnings

import numpy as np

warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from data.utils import *


def reparameterization(data_sample, history_sample):
    # reparameterization the features to let if follow the constraint of
    # low<=open<=close<=high
    # old features: open,high,low,close
    # new features: open, O_minus_L,C_minus_L,H_minus_maxOC
    # O_minus_L: open-low
    # C_minus_L: close-low
    # H_minus_maxOC: high-max(open,close)
    # these features should be non-negative
    O = data_sample['open']
    H = data_sample['high']
    L = data_sample['low']
    C = data_sample['close']
    O_minus_L = O - L
    C_minus_L = C - L
    H_minus_maxOC = H - np.maximum(O, C)
    # insert these features into the data_sample
    data_sample.insert(1, 'O_minus_L', O_minus_L)
    data_sample.insert(2, 'C_minus_L', C_minus_L)
    data_sample.insert(3, 'H_minus_maxOC', H_minus_maxOC)
    # do the same thing to history_sample
    O_h = history_sample['open']
    H_h = history_sample['high']
    L_h = history_sample['low']
    C_h = history_sample['close']
    O_minus_L_h = O_h - L_h
    C_minus_L_h = C_h - L_h
    H_minus_maxOC_h = H_h - np.maximum(O_h, C_h)
    # insert these features into the history_sample
    history_sample.insert(1, 'O_minus_L', O_minus_L_h)
    history_sample.insert(2, 'C_minus_L', C_minus_L_h)
    history_sample.insert(3, 'H_minus_maxOC', H_minus_maxOC_h)

    # drop the 'high','open','close' feature
    data_sample = data_sample.drop(['high', 'open', 'close'], axis=1)
    history_sample = history_sample.drop(['high', 'open', 'close'], axis=1)

    # let the column follow the order of 'low','O_minus_L','C_minus_L','H_minus_maxOC'
    data_sample = data_sample[['low', 'O_minus_L', 'C_minus_L', 'H_minus_maxOC']]
    history_sample = history_sample[['low', 'O_minus_L', 'C_minus_L', 'H_minus_maxOC']]

    return data_sample, history_sample


def normalization(data_sample, history_sample):
    # normalize the data_sample and history_sample based on normalization of history_sample
    # for the 'low' we use the MinMaxScaler from range(-1,1)
    # for the 'O_minus_L','C_minus_L','H_minus_maxOC' we project them to have a range of (0,1) with the max value as multiplier
    # this is to avoid the negative values of these features in the data_sample

    # normalize the 'low' feature
    low_scaler = MinMaxScaler(feature_range=(-1, 1))
    low_scaler.fit(history_sample['low'].values.reshape(-1, 1))
    data_sample['low'] = low_scaler.transform(data_sample['low'].values.reshape(-1, 1))
    history_sample['low'] = low_scaler.transform(history_sample['low'].values.reshape(-1, 1))
    # normalize the 'O_minus_L','C_minus_L','H_minus_maxOC' features
    O_minus_L_scaler = 1 / np.max(history_sample['O_minus_L'])
    C_minus_L_scaler = 1 / np.max(history_sample['C_minus_L'])
    H_minus_maxOC_scaler = 1 / np.max(history_sample['H_minus_maxOC'])
    data_sample['O_minus_L'] = data_sample['O_minus_L'] * O_minus_L_scaler
    data_sample['C_minus_L'] = data_sample['C_minus_L'] * C_minus_L_scaler
    data_sample['H_minus_maxOC'] = data_sample['H_minus_maxOC'] * H_minus_maxOC_scaler
    history_sample['O_minus_L'] = history_sample['O_minus_L'] * O_minus_L_scaler
    history_sample['C_minus_L'] = history_sample['C_minus_L'] * C_minus_L_scaler
    history_sample['H_minus_maxOC'] = history_sample['H_minus_maxOC'] * H_minus_maxOC_scaler

    return data_sample, history_sample, low_scaler, O_minus_L_scaler, C_minus_L_scaler, H_minus_maxOC_scaler


def normalization_inverse(data_sample, scaler_dict, scaler_order, feature_order):
    low_scaler = scaler_dict[:, scaler_order.index('low')]
    O_minus_L_scaler = scaler_dict[:, scaler_order.index('O_minus_L')]
    C_minus_L_scaler = scaler_dict[:, scaler_order.index('C_minus_L')]
    H_minus_maxOC_scaler = scaler_dict[:, scaler_order.index('H_minus_maxOC')]

    low_index = feature_order.index('low')
    O_minus_L_index = feature_order.index('O_minus_L')
    C_minus_L_index = feature_order.index('C_minus_L')
    H_minus_maxOC_index = feature_order.index('H_minus_maxOC')

    # inverse the normalization of 'low'
    # we don't have column name for the scaler, so we use the index to get the scaler
    for sample in range(data_sample.shape[0]):
        data_sample[sample, :, low_index] = low_scaler[sample].inverse_transform(
            data_sample[sample, :, low_index].reshape(-1, 1)).reshape(-1)
        # inverse the normalization of 'O_minus_L','C_minus_L','H_minus_maxOC'
        data_sample[sample, :, O_minus_L_index] = data_sample[sample, :, O_minus_L_index] / O_minus_L_scaler[sample]
        data_sample[sample, :, C_minus_L_index] = data_sample[sample, :, C_minus_L_index] / C_minus_L_scaler[sample]
        data_sample[sample, :, H_minus_maxOC_index] = data_sample[sample, :, H_minus_maxOC_index] / \
                                                      H_minus_maxOC_scaler[sample]

    return data_sample


def reparameterization_inverse(data, original_feature_order, feature_order):
    # inverse process of reparameterization
    low_index = feature_order.index('low')
    O_minus_L_index = feature_order.index('O_minus_L')
    C_minus_L_index = feature_order.index('C_minus_L')
    H_minus_maxOC_index = feature_order.index('H_minus_maxOC')

    # get OHLC based on the reparameterized data
    open = data[:, :, low_index] + data[:, :, O_minus_L_index]
    close = data[:, :, low_index] + data[:, :, C_minus_L_index]
    high = np.maximum(open, close) + data[:, :, H_minus_maxOC_index]

    # assign the OHLC to the data according to the order in the feature
    low_i = original_feature_order.index('low')
    open_i = original_feature_order.index('open')
    close_i = original_feature_order.index('close')
    high_i = original_feature_order.index('high')

    data[:, :, low_i] = data[:, :, low_index]
    data[:, :, open_i] = open
    data[:, :, close_i] = close
    data[:, :, high_i] = high
    return data


def conditional_data_sample_worker(data, seq_len, features, history_length, full_data, differential_features=True):
    time = []
    intervals = get_intervals(data)

    normalized_data_samples = []
    normalized_history_samples = []
    original_data_samples = []
    original_history_samples = []
    low_scaler_vec = []
    O_minus_L_scaler_vec = []
    C_minus_L_scaler_vec = []
    H_minus_maxOC_scaler_vec = []
    # get the starting index of each interval
    start_index_by_interval = []
    last_hist_vec = []
    index = data['index']
    for interval in intervals:
        start_index_by_interval.append(index[interval[0]])

    for interval_index, interval in enumerate(intervals):
        data_seg = data.iloc[interval[0]:interval[1], :]
        data_seg = data_seg.loc[:, features]
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

            # reparameterization the data_sample and history_sample
            check_data_processing = True
            if check_data_processing == True:
                ori_data_sample = data_sample.copy()
                ori_hitory_sample = history_sample.copy()
            else:
                ori_data_sample = None
                ori_hitory_sample = None
            data_sample, history_sample = reparameterization(data_sample, history_sample)

            # differential features on 'low' for data_sample and history_sample

            try:
                last_low_befor_data_sample = data.iloc[interval[0] + i - 1, :]['low']
                # do the same thing to history_sample
                last_low_befor_history_sample = full_data.iloc[history_start - 1, :]['low']
                if differential_features:
                    data_sample_low_diff = data_sample['low'].diff()
                    # the first element of data_sample_low_diff is nan, we replace it with the first low in data sample minus last_low_befor_data_sample
                    data_sample_low_diff.iloc[0] = data_sample['low'].iloc[0] - last_low_befor_data_sample
                    data_sample['low'] = data_sample_low_diff
                    # do the same thing to history_sample
                    history_sample_low_diff = history_sample['low'].diff()
                    history_sample_low_diff.iloc[0] = history_sample['low'].iloc[0] - last_low_befor_history_sample
                    history_sample['low'] = history_sample_low_diff
                last_hist_vec.append(np.array([last_low_befor_data_sample, last_low_befor_history_sample]))
            except:
                continue

            # normalization
            # we use the 'low' feature of historical data to normalize the data
            normalized_data_sample, normalized_history_sample, low_scaler, O_minus_L_scaler, C_minus_L_scaler, H_minus_maxOC_scaler = normalization(
                data_sample, history_sample)
            low_scaler_vec.append(low_scaler)
            O_minus_L_scaler_vec.append(O_minus_L_scaler)
            C_minus_L_scaler_vec.append(C_minus_L_scaler)
            H_minus_maxOC_scaler_vec.append(H_minus_maxOC_scaler)
            time.append(seq_len)
            normalized_data_samples.append(normalized_data_sample)
            normalized_history_samples.append(normalized_history_sample)
            original_data_samples.append(ori_data_sample)
            original_history_samples.append(ori_hitory_sample)
    return normalized_data_samples, normalized_history_samples, time, low_scaler_vec, O_minus_L_scaler_vec, C_minus_L_scaler_vec, H_minus_maxOC_scaler_vec, last_hist_vec, original_data_samples, original_history_samples


def conditional_dataset_preprocessing(data_file_path, tics, args, tic_tokenizer, dynamics_tokenizer, features,
                                      history_length):
    # 1.get the data by stock/dynamics type
    # 2. get data sample (and history) of the same length
    # 3. StandardScaler the data by its own history
    # 4. Re-parameterization the data according to the constraint

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

    low_scaler_vec_list = []
    O_minus_L_scaler_vec_list = []
    C_minus_L_scaler_vec_list = []
    H_minus_maxOC_scaler_vec_list = []

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
            data_samples, historcial_samples, time, low_scaler_vec, O_minus_L_scaler_vec, C_minus_L_scaler_vec, H_minus_maxOC_scaler_vec, last_hist_vec, original_data_samples, original_history_samples = conditional_data_sample_worker(
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
            low_scaler_vec_list.extend(low_scaler_vec)
            O_minus_L_scaler_vec_list.extend(O_minus_L_scaler_vec)
            C_minus_L_scaler_vec_list.extend(C_minus_L_scaler_vec)
            H_minus_maxOC_scaler_vec_list.extend(H_minus_maxOC_scaler_vec)

    # stack all data slices
    # concatenate X_list
    X = np.concatenate(X_list, axis=0)  # dim: (n,seq_len,features)
    H = np.concatenate(H_list, axis=0)  # dim: (n,historical_len,features)
    T = T_list  # dim: (n)
    D = np.vstack(D_list)  # dim: (n,dynmamic_token)
    L = np.vstack(L_list)  # dim: (n,tic_token)
    OX = np.concatenate(OX_list, axis=0)  # dim: (n,seq_len,features)
    OH = np.concatenate(OH_list, axis=0)  # dim: (n,historical_len,features)
    low_scaler = low_scaler_vec_list  # dim: (n)
    O_minus_L_scaler = O_minus_L_scaler_vec_list  # dim: (n)
    C_minus_L_scaler = C_minus_L_scaler_vec_list  # dim: (n)
    H_minus_maxOC_scaler = H_minus_maxOC_scaler_vec_list  # dim: (n)
    Last_h = np.vstack(Last_h_list)  # dim: (n,2)
    # shape and dtype of X
    print("X.shape: ", X.shape, " X.dtype: ", X.dtype)
    print('H.shape: ', H.shape, ' H.dtype: ', H.dtype)
    print("T.shape: ", np.array(T).shape, " T.dtype: ", np.array(T).dtype)
    print("D.shape: ", D.shape, " D.dtype: ", D.dtype)
    print("L.shape: ", L.shape, " L.dtype: ", L.dtype)
    print("Last_h.shape: ", Last_h.shape, " Last_h.dtype: ", Last_h.dtype)
    # stack the scalr list to a scaler vector for each sample
    scaler = np.vstack([low_scaler, O_minus_L_scaler, C_minus_L_scaler, H_minus_maxOC_scaler]).T  # dim: (n,4)
    print("scaler.shape: ", scaler.shape, " scaler.dtype: ", scaler.dtype)
    # save the order of the scaler
    scaler_order = ['low', 'O_minus_L', 'C_minus_L', 'H_minus_maxOC']

    return X, H, T, D, L, scaler, scaler_order, Last_h, OX, OH


def conditional_rescale_data(ori_data, scaler, differential_features, last_hist_value, scaler_order,
                             original_feature_order):
    # the last_hist_value is the last value of the historical data
    # if differential_features, then use the last_hist_value to reconstruct the original data
    # first rescale the data, then reconstruct the original data
    # reconstruction step: 1.reconstruct the low 2. reconstruct the rest features

    feature_order = ['low', 'O_minus_L', 'C_minus_L', 'H_minus_maxOC']
    data = ori_data.copy()
    # rescale the data
    data = normalization_inverse(data, scaler, scaler_order, feature_order)
    # reconstruct the 'low' feature if differential_features
    print(last_hist_value[:2])
    if differential_features:
        low_index = feature_order.index('low')
        # reconstruct the low feature by adding the cumulative sum of the low feature start from the last_hist_value
        data[:, :, low_index] = np.cumsum(data[:, :, low_index], axis=1) + last_hist_value
    # reverse the re-parameterization
    data = reparameterization_inverse(data, original_feature_order=original_feature_order, feature_order=feature_order)

    return data
