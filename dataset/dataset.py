import numpy as np
import torch


class FeaturePredictionDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    - idx (int): the index of the feature to be predicted
    """

    def __init__(self, data, time, idx):
        no, seq_len, dim = data.shape
        self.X = torch.FloatTensor(
            np.concatenate(
                (data[:, :, :idx], data[:, :, (idx + 1):]),
                axis=2
            )
        )
        self.T = torch.LongTensor(time)
        self.Y = torch.FloatTensor(np.reshape(data[:, :, idx], [no, seq_len, 1]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]


class OneStepPredictionDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, data, time, if_Timesnet):
        # Timesnet will use another datastructure
        if if_Timesnet:
            self.X = torch.FloatTensor(data[:, :-1, :])
            self.T = torch.LongTensor([t - 1 if t == data.shape[1] else t for t in time])
            self.Y = torch.FloatTensor(data[:, -1, :])
            # self.Y shape from (B, F) to (B,1,F)
            self.Y = self.Y.unsqueeze(1)
        else:
            self.X = torch.FloatTensor(data[:, :-1, :])
            self.T = torch.LongTensor([t - 1 if t == data.shape[1] else t for t in time])
            self.Y = torch.FloatTensor(data[:, 1:, :])
            self.Y = self.Y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]


class DiscriminatorDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, ori_data, generated_data, ori_time, generated_time):
        self.Fake_data = torch.FloatTensor(generated_data)
        self.Fake_t = torch.LongTensor(generated_time)
        self.Real_t = torch.LongTensor(ori_time)
        self.Real_data = torch.FloatTensor(ori_data)

    def __len__(self):
        return len(self.Fake_data)

    def __getitem__(self, idx):
        return self.Fake_data[idx], self.Fake_t[idx], self.Real_data[idx], self.Real_t[idx]


class PredictorDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, ori_data, generated_data, ori_time, generated_time):
        self.X = torch.FloatTensor(generated_data)
        self.T = torch.LongTensor(ori_time)
        self.Y = torch.FloatTensor(ori_data)
        self.Y_T = torch.LongTensor(generated_time)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx], self.Y_T[idx]


class LabelPredictionDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, data, time, label):
        self.X = torch.FloatTensor(data)
        self.T = torch.LongTensor(time)
        self.Y = torch.FloatTensor(label)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]


class PosthocDiscriminatorDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, real_data, real_time, real_label, fake_data, fake_time, fake_label):
        self.X_real = torch.FloatTensor(real_data)
        self.T_real = torch.LongTensor(real_time)
        self.Y_real = torch.FloatTensor(real_label)
        self.X_fake = torch.FloatTensor(fake_data)
        self.T_fake = torch.LongTensor(fake_time)
        self.Y_fake = torch.FloatTensor(fake_label)

    def __len__(self):
        return len(self.X_real)

    def __getitem__(self, idx):
        return self.X_real[idx], self.T_real[idx], self.Y_real[idx], self.X_fake[idx], self.T_fake[idx], self.Y_fake[
            idx]


class TimeGANDataset(torch.utils.data.Dataset):
    """TimeGAN Dataset for sampling data with their respective time

    Args:
        - data (numpy.ndarray): the padded dataset to be fitted (D x S x F)
        - time (numpy.ndarray): the length of each data (D)
    Parameters:
        - x (torch.FloatTensor): the real value features of the data
        - t (torch.LongTensor): the temporal feature of the data
    """

    def __init__(self, data, time=None, padding_value=None):
        # sanity check
        if len(data) != len(time):
            raise ValueError(
                f"len(data) `{len(data)}` != len(time) {len(time)}"
            )

        if isinstance(time, type(None)):
            time = [len(x) for x in data]

        self.X = torch.FloatTensor(data)
        self.T = torch.LongTensor(time)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]

        # The actual length of each data
        T_mb = [T for T in batch[1]]

        return X_mb, T_mb


class ConditionalTimeGANDataset(torch.utils.data.Dataset):
    """TimeGAN Dataset for sampling data with their respective time

    Args:
        - data (numpy.ndarray): the padded dataset to be fitted (D x S x F)
        - time (numpy.ndarray): the length of each data (D)
        - dynamic (numpy.ndarray): the one-hot dynamic feature of each data (D x Dim_D)
        - label (numpy.ndarray): the one-hot label of each data (D x Dim_L)
        - history (numpy.ndarray): the history of each data (D x history_length x F)

    Parameters:
        - x (torch.FloatTensor): the real value features of the data
        - t (torch.LongTensor): the temporal feature of the data
    """

    def __init__(self, data, time=None, padding_value=None, dynamic=None, label=None, history=None):
        # sanity check
        if len(data) != len(time):
            raise ValueError(
                f"len(data) `{len(data)}` != len(time) {len(time)}"
            )

        if isinstance(time, type(None)):
            time = [len(x) for x in data]
        if isinstance(dynamic, type(None)):
            dynamic = [None for _ in data]
            self.D = dynamic
        else:
            self.D = torch.FloatTensor(dynamic)
        if isinstance(label, type(None)):
            label = [None for _ in data]
            self.L = label
        else:
            self.L = torch.FloatTensor(label)
        if isinstance(history, type(None)):
            history = [None for _ in data]
            self.H = history
        else:
            self.H = torch.FloatTensor(history)
        self.X = torch.FloatTensor(data)
        self.T = torch.LongTensor(time)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.D[idx], self.L[idx], self.H[idx]

    def collate_fn(self, batch):
        """Minibatch sampling
        """
        # Pad sequences to max length
        X_mb = [X for X in batch[0]]

        # The actual length of each data
        T_mb = [T for T in batch[1]]

        D_mb = [D for D in batch[2]]
        L_mb = [L for L in batch[3]]

        H_mb = [H for H in batch[4]]

        return X_mb, T_mb, D_mb, L_mb, H_mb
