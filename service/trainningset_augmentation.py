import argparse
import os
import pickle
import random

import numpy as np
import torch

from MarketGAN_service import MarketGAN_service

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./output/DJ30_V2_RT', help='path to the model')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--result_path', type=str, default='./downstream_tasks/data/downstream_tasks/data/MarketGAN',
                        help='path to the result')
    parser.add_argument('--inference_sample_folder', type=str,
                        default='./downstream_tasks/data/downstream_tasks/data/processed',
                        help='path to the inference sample')
    parser.add_argument('--device', type=str, default='3', help='device to use')
    args = parser.parse_args()
    print('model_path: ', args.model_path)
    print('result_path: ', args.result_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    service = MarketGAN_service(args.model_path, 'cuda:' + args.device)
    # get the data from inference_sample_folder
    for file in os.listdir(args.inference_sample_folder):
        # read the data from the file with pickle
        if file.startswith('data_sample_train'):
            with open(os.path.join(args.inference_sample_folder, file), 'rb') as f:
                data = pickle.load(f)
            # parse the ticker name and dynamic label from the file name
            ticker_name = file.split('_')[3]
            dynamic_label = file.split('_')[4].split('.')[0]

            H = data['history']
            scaler = data['scaler']
            Last_h = data['last_hist_vec']
            H = np.array(H, dtype=float)
            scaler = np.array(scaler)
            Last_h = np.array(Last_h, dtype=float)
            time = service.args.max_seq_len
            time = np.array([time] * H.shape[0])

            generated_X_rescaled = service.inference(dynamic_label, ticker_name, H, scaler, Last_h, time)
            # save the generated data to the file
            with open(os.path.join(args.result_path, 'generated_data_' + ticker_name + '_' + dynamic_label + '.pkl'),
                      'wb') as f:
                pickle.dump(generated_X_rescaled, f)
