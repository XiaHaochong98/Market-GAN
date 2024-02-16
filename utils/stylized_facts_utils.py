import argparse
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


def stylized_facts(df, verbose=False):
    # Assuming `returns` is your series of returns and `volume` is the trading volume
    returns = df['return']  # replace with your returns data
    # volume = df['Volume'] # replace with your volume data

    # Autocorrelation of returns
    returns_acf = acf(returns)[1:]  # change nlags as needed
    returns_acf = returns_acf.mean()
    if verbose:
        print("Autocorrelation of returns:", returns_acf)

    # Autocorrelation of absolute returns
    abs_returns_acf = acf(np.abs(returns))[1:]  # change nlags as needed
    abs_returns_acf = abs_returns_acf.mean()
    if verbose:
        print("Autocorrelation of absolute returns:", abs_returns_acf)

    # Leverage effect (negative correlation between returns and future volatility)
    volatility = returns.rolling(window=10).std()  # change window size as needed
    leverage_corr = np.corrcoef(returns[:-10], volatility[10:])[0, 1]
    if verbose:
        print("Leverage effect:", leverage_corr)
    # Correlation with volume
    # volume_return_corr = np.corrcoef(volume, np.abs(returns))[0, 1]
    # if verbose:
    #     print("Volume-return correlation:", volume_return_corr)
    return returns_acf, abs_returns_acf, leverage_corr


def stylized_facts_by_chunks(chunks, features, verbose=False):
    returns_acf_list = []
    abs_returns_acf_list = []
    leverage_corr_list = []
    volume_return_corr_list = []
    for chunk in chunks:
        # convert chunk to df with columns names as features
        df = pd.DataFrame(chunk, columns=features)
        df['return'] = df.close / df.close.shift(1)
        # df = df[['adj_close', 'log_rtn']].dropna(how = 'any')
        df = df.dropna(how='any')
        returns_acf, abs_returns_acf, leverage_corr = stylized_facts(df)
        returns_acf_list.append(returns_acf)
        abs_returns_acf_list.append(abs_returns_acf)
        leverage_corr_list.append(leverage_corr)
        # volume_return_corr_list.append(volume_return_corr)
    # use pd.describe the returns as pd dataframe
    returns_acf_df = pd.DataFrame(returns_acf_list)
    abs_returns_acf_df = pd.DataFrame(abs_returns_acf_list)
    leverage_corr_df = pd.DataFrame(leverage_corr_list)
    # volume_return_corr_df = pd.DataFrame(volume_return_corr_list)
    if verbose:
        print(returns_acf_df.describe())
        print(abs_returns_acf_df.describe())
        print(leverage_corr_df.describe())
        # print(volume_return_corr_df.describe())
    # return a dict of dataframes
    return {'returns_acf': returns_acf_df, 'abs_returns_acf': abs_returns_acf_df,
            'leverage_corr': leverage_corr_df}


def stylized_facts_by_file(path, verbose=False):
    returns_acf_list = []
    abs_returns_acf_list = []
    leverage_corr_list = []
    volume_return_corr_list = []

    if os.path.isdir(path):
        for file in os.listdir(path):
            df = pd.read_csv(os.path.join(path, file))
            # all columns are in lower case
            df.columns = map(str.lower, df.columns)
            df['return'] = df.close / df.close.shift(1)
            # df = df[['adj_close', 'log_rtn']].dropna(how = 'any')
            df = df.dropna(how='any')
            returns_acf, abs_returns_acf, leverage_corr, volume_return_corr = stylized_facts(df)
            returns_acf_list.append(returns_acf)
            abs_returns_acf_list.append(abs_returns_acf)
            leverage_corr_list.append(leverage_corr)
            volume_return_corr_list.append(volume_return_corr)
    # use pd.describe the returns as pd dataframe
    returns_acf_df = pd.DataFrame(returns_acf_list)
    abs_returns_acf_df = pd.DataFrame(abs_returns_acf_list)
    leverage_corr_df = pd.DataFrame(leverage_corr_list)
    volume_return_corr_df = pd.DataFrame(volume_return_corr_list)
    if verbose:
        print(returns_acf_df.describe())
        print(abs_returns_acf_df.describe())
        print(leverage_corr_df.describe())
        print(volume_return_corr_df.describe())


def stylized_facts_by_tic(df, verbose=False):
    returns_acf_list = []
    abs_returns_acf_list = []
    leverage_corr_list = []
    volume_return_corr_list = []
    for tic in df.tic.unique():
        if verbose:
            print(f'tic is {tic}')
        df_tic = df[df['tic'] == tic]
        # all columns are in lower case
        df_tic['return'] = df_tic.close / df_tic.close.shift(1)
        # df = df[['adj_close', 'log_rtn']].dropna(how = 'any')
        df_tic = df_tic.dropna(how='any')
        returns_acf, abs_returns_acf, leverage_corr, volume_return_corr = stylized_facts(df_tic)
        returns_acf_list.append(returns_acf)
        abs_returns_acf_list.append(abs_returns_acf)
        leverage_corr_list.append(leverage_corr)
        volume_return_corr_list.append(volume_return_corr)
    # use pd.describe the returns as pd dataframe
    returns_acf_df = pd.DataFrame(returns_acf_list)
    abs_returns_acf_df = pd.DataFrame(abs_returns_acf_list)
    leverage_corr_df = pd.DataFrame(leverage_corr_list)
    volume_return_corr_df = pd.DataFrame(volume_return_corr_list)
    if verbose:
        print(returns_acf_df.describe())
        print(abs_returns_acf_df.describe())
        print(leverage_corr_df.describe())
        print(volume_return_corr_df.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='')
    parser.add_argument("--mode", type=str, default='tic')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    # log output to file
    if not os.path.exists('stylized_facts_log'):
        os.makedirs('stylized_facts_log')
    f = open(f"stylized_facts_log/stylized_facts_{args.path}.log", 'a')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    if args.mode == 'tic':
        stylized_facts_by_tic(args.path, args.verbose)
    elif args.mode == 'file':
        stylized_facts_by_file(args.path, args.verbose)
    else:
        raise NotImplementedError
