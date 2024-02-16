## Necessary packages

import warnings

import numpy as np

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
    # print(low_scaler,O_minus_L_scaler)
    # for each sample, we need to inverse the normalization of 'low'
    # print(data_sample.shape)
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


def data_preprocessing_for_condition_discrimination(history_data_rescaled, generated_data_samples, last_hist_vec,
                                                    features, differential_features=True):
    # processed the data into encoding format for condition discriminator

    normalized_data_samples = []
    for index, generated_data_sample in enumerate(generated_data_samples):
        history_sample = history_data_rescaled[index]
        # cast the history_sample and generated_data_sample to dataframe with columns as features
        history_sample = pd.DataFrame(history_sample, columns=features)
        ori_history_sample = history_sample.copy()
        generated_data_sample = pd.DataFrame(generated_data_sample, columns=features)
        data_sample, history_sample = reparameterization(generated_data_sample, history_sample)
        # differential features on 'low' for data_sample and history_sample
        try:
            last_low_befor_data_sample = ori_history_sample.iloc[-1, :]['low']
            last_low_befor_history_sample = last_hist_vec[index]
            if differential_features:
                data_sample_low_diff = data_sample['low'].diff()
                # the first element of data_sample_low_diff is nan, we replace it with the first low in data sample minus last_low_befor_data_sample
                data_sample_low_diff.iloc[0] = data_sample['low'].iloc[0] - last_low_befor_data_sample
                data_sample['low'] = data_sample_low_diff
                history_sample_low_diff = history_sample['low'].diff()
                history_sample_low_diff.iloc[0] = history_sample['low'].iloc[0] - last_low_befor_history_sample
                history_sample['low'] = history_sample_low_diff
        except:
            continue

        # normalization
        # we use the 'low' feature of historical data to normalize the data
        normalized_data_sample, normalized_history_sample, low_scaler, O_minus_L_scaler, C_minus_L_scaler, H_minus_maxOC_scaler = normalization(
            data_sample, history_sample)
        normalized_data_samples.append(normalized_data_sample)
        # cast the normalized_data_samples to numpy array
    normalized_data_samples = np.array(normalized_data_samples)
    return normalized_data_samples


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
