import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_intervals(data):
    index = data['index']
    last_value = index[0] - 1
    last_index = 0
    intervals = []
    for i in range(data.shape[0]):
        if last_value != index[i] - 1:
            intervals.append([last_index, i])
            last_value = index[i]
            last_index = i
        last_value = index[i]
    intervals.append([last_index, i])
    return intervals


def interpolation(data):
    max_len = 24
    l = len(data)
    to_fill = max_len - l
    if to_fill != 0:
        interval = max_len // to_fill
        for j in range(to_fill):
            idx = (interval + 1) * j + interval
            data.insert(min(idx, len(data) - 1), float('nan'))
    data = pd.Series(data).interpolate(method='polynomial', order=2)
    return data


def get_tics(path):
    data = pd.read_csv(path).reset_index()
    tics = data['tic'].unique()
    return tics


def train_test_split_by_conditions(args, dynamics_tokenizer, tic_tokenizer, X, T, D, L, scaler, H, Last_h, OX, OH):
    train_X = []
    test_X = []
    train_time = []
    test_time = []
    train_D = []
    test_D = []
    train_L = []
    test_L = []
    train_H = []
    test_H = []
    train_Last_h = []
    test_Last_h = []
    train_scaler = []
    test_scaler = []
    OX_train = []
    OX_test = []
    OH_train = []
    OH_test = []

    for dynamic in range(args.dynamic_dim):
        for tic in args.tics:
            # slice the data for each dynamic and tic
            D_word = dynamics_tokenizer.one_hot_to_word(D)
            L_word = tic_tokenizer.one_hot_to_word(L)

            indexs = []
            for i in range(len(D_word)):
                if D_word[i] == str(dynamic) and L_word[i] == str(tic):
                    indexs.append(i)
            # train_index = np.array(train_index)
            indexs = np.array(indexs)

            data_slice = X[indexs]
            time_slice = list(np.array(T)[indexs])
            D_slice = D[indexs]
            L_slice = L[indexs]
            H_slice = H[indexs]
            Last_h_slice = Last_h[indexs]
            OX_slice = OX[indexs]
            OH_slice = OH[indexs]
            scaler_slice = scaler[indexs]
            # split the slices into train and test
            train_data_slice, test_data_slice, train_time_slice, test_time_slice, train_D_slice, test_D_slice, train_L_slice, test_L_slice, train_scaler_slice, test_scaler_slice, train_H_slice, test_H_slice, train_Last_h_slice, test_Last_h_slice, OX_train_slice, OX_test_slice, OH_train_slice, OH_test_slice = train_test_split(
                data_slice, time_slice, D_slice, L_slice, scaler_slice, H_slice, Last_h_slice, OX_slice, OH_slice,
                test_size=1 - args.train_rate, random_state=args.seed
            )
            # append the slices to the lists
            train_X.append(train_data_slice)
            test_X.append(test_data_slice)
            train_time.append(train_time_slice)
            test_time.append(test_time_slice)
            train_D.append(train_D_slice)
            test_D.append(test_D_slice)
            train_L.append(train_L_slice)
            test_L.append(test_L_slice)
            train_H.append(train_H_slice)
            test_H.append(test_H_slice)
            train_Last_h.append(train_Last_h_slice)
            test_Last_h.append(test_Last_h_slice)
            train_scaler.append(train_scaler_slice)
            test_scaler.append(test_scaler_slice)
            OX_train.append(OX_train_slice)
            OX_test.append(OX_test_slice)
            OH_train.append(OH_train_slice)
            OH_test.append(OH_test_slice)
            print(f'dynamic {dynamic} tic {tic} train sample number {train_data_slice.shape}')
            print(f'dynamic {dynamic} tic {tic} test sample number {test_data_slice.shape}')
    # concatenate the lists in to arrays
    train_X = np.concatenate(train_X, axis=0)
    test_X = np.concatenate(test_X, axis=0)
    train_time = np.concatenate(train_time, axis=0)
    test_time = np.concatenate(test_time, axis=0)
    train_D = np.concatenate(train_D, axis=0)
    test_D = np.concatenate(test_D, axis=0)
    train_L = np.concatenate(train_L, axis=0)
    test_L = np.concatenate(test_L, axis=0)
    train_H = np.concatenate(train_H, axis=0)
    test_H = np.concatenate(test_H, axis=0)
    train_Last_h = np.concatenate(train_Last_h, axis=0)
    test_Last_h = np.concatenate(test_Last_h, axis=0)
    OX_train = np.concatenate(OX_train, axis=0)
    OX_test = np.concatenate(OX_test, axis=0)
    OH_train = np.concatenate(OH_train, axis=0)
    OH_test = np.concatenate(OH_test, axis=0)
    train_scaler = np.concatenate(train_scaler, axis=0)
    test_scaler = np.concatenate(test_scaler, axis=0)

    return train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler, train_H, test_H, train_Last_h, test_Last_h, OX_train, OX_test, OH_train, OH_test
