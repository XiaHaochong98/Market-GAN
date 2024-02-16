# train prediction model with different training sets
# set system path to root folder
import sys
import argparse
import os
import torch
import random
sys.path.append(os.path.abspath('../'))
from metrics.metric_utils import one_step_ahead_prediction
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import datetime
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default='helloworld')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prediction_model', type=str, default='TimesNet')
    args=parser.parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    baseline_model_list=['SigCWGAN','TimeGAN','GMMN','RCGAN']

    # prepare the training set according to the experiment

    if args.device.startswith("cuda") and torch.cuda.is_available():
        print("Using CUDA\n")
        try:
            # os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
            args.device = torch.device(f"{args.device}")
            print(f"Using device {args.device}.\n")
        except:
            args.device = torch.device("cuda:0")
            print(f"Invalid device name {args.device}. Using cuda:0 instead.\n")

    experiment_folder = f'./experiments_results/{args.exp_name}'
    # create experiment folder if not exist
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    experiment_result_file=os.path.join(experiment_folder, 'result.csv')
    task=args.task
    real_data_folder='./data/downstream_tasks/data/original'
    benchmark_folder='./data/downstream_tasks/original_result/DJ30_downstream_inference'
    benchmarket_encoding_folder='./data/downstream_tasks/processed_result_encoded/DJ30_downstream_inference'
    MarketGAN_folder='./data/downstream_tasks/data/MarketGAN'
    MarketGAN_RNN_folder='./data/downstream_tasks/data/MarketGAN_RNN'
    # read in the real training set to get the tics and dynamics in the training set
    # the data file templates is 'data_sample_{task}_{}_{}.pkl'.format(task, tic, dynamic)
    # example 'data_sample_test_AAPL_0.pkl'
    # parse all the tic and dynamics in the training set
    tics = set()
    dynamics = set()
    for file in os.listdir(real_data_folder):
        if file.endswith('.pkl'):
            tic = file.split('_')[3]
            D = file.split('_')[4].split('.')[0]
            # print('tic:', tic, 'D:', D)
            tics.add(tic)
            dynamics.add(D)
    print('tics number:', len(tics))
    print('dynamics number:', len(dynamics))
    print('batch_size:', args.batch_size)

    # decide the model list according to the experiment
    if task=='benchmark' or task=='benchmark_encoding':
        model_list=baseline_model_list
    elif task=='MarketGAN':
        model_list=['MarketGAN']
    elif task=='MarketGAN_RNN':
        model_list = ['MarketGAN_RNN']
    elif task=='real':
        model_list=['real']
    else:
        raise ValueError('exp should be one of benchmarks,benchmarks_encoding,MarketGAN,MarketGAN_RNN,real')



    #prepare the training set according to the experiment
    for model in tqdm(model_list, desc="model", position=0):
        model_result = pd.DataFrame(columns=[
            'model_name','experiment_time','pred_SMAPE', 'pred_norm_SMAPE', 'trainset_MSE',
            'avg_pred_SMAPE', 'pred_SMAPE_std', 'pred_norm_SMAPE_std',
            'trainset_MSE_std', 'pred_SMAPE_detail', 'pred_norm_SMAPE_detail',
            'trainset_MSE_detail', 'avg_pred_SMAPE_detail',
        ])
        pred_SMAPE_list = []
        pred_norm_SMAPE_list = []
        avg_pred_SMAPE_list = []
        trainset_MSE_list=[]

        for D in tqdm(dynamics, desc="dynamics", position=1, leave=False):
            for tic in tqdm(tics, desc="tic", position=2, leave=False):

                print('model:',model, 'tic:', tic, 'D:', D)

                # read in the real training set
                with open(os.path.join(real_data_folder, 'data_sample_{}_{}_{}.pkl'.format('train', tic, D)), 'rb') as f:
                    # keys ['data', 'time', 'scaler', 'last_hist_vec', 'history']
                    real_train_dict = pickle.load(f)
                    real_train=real_train_dict['data']
                    # cast real_train to array
                    real_train=np.array(real_train)
                    train_time=real_train_dict['time']
                    epochs_sum=real_train.shape[0]*args.epoch
                # read in the real test set
                with open(os.path.join(real_data_folder, 'data_sample_{}_{}_{}.pkl'.format('test', tic, D)), 'rb') as f:
                    real_test_dict = pickle.load(f)
                    real_test=real_test_dict['data']
                if task=='benchmark':
                    file_path=os.path.join(benchmark_folder, f'{tic}_{D}/seed=0/{model}/{model}_{tic}_{D}_generated_test.npy')
                    print('augmentation file_path',file_path)
                    generated_data_as_train = np.load(file_path)
                elif task=='benchmark_encoding':
                    file_path = os.path.join(benchmarket_encoding_folder, f'{tic}_{D}/seed=0/{model}/{model}_{tic}_{D}_generated_test.npy')
                    print('augmentation file_path', file_path)
                    generated_data_as_train = np.load(file_path)
                elif task=='MarketGAN':
                    file_path=os.path.join(MarketGAN_folder, f'generated_data_{tic}_{D}.pkl')
                    print('augmentation file_path', file_path)
                    # read pickle file
                    generated_data_as_train = pickle.load(open(file_path, 'rb'))
                elif task=='MarketGAN_RNN':
                    file_path=os.path.join(MarketGAN_RNN_folder, f'generated_data_{tic}_{D}.pkl')
                    print('augmentation file_path', file_path)
                    # read pickle file
                    generated_data_as_train = pickle.load(open(file_path, 'rb'))
                elif task=='real':
                    # generated_data_as_train is a empty np array
                    print('augmentation file_path', 'None')
                    generated_data_as_train=np.array([])
                else:
                    raise ValueError('exp should be one of benchmarks,benchmarks_encoding,MarketGAN')

                # combine the real training set and generated training set
                if generated_data_as_train.shape[0]==0:
                    train_set=real_train
                else:
                    print('real_train.shape',real_train.shape)
                    print('generated_data_as_train.shape',generated_data_as_train.shape)
                    print('generated_data_as_train preview',generated_data_as_train[0:2,0:2,:])
                    train_set=np.concatenate((real_train,generated_data_as_train),axis=0)
                    train_time=np.concatenate((train_time,train_time),axis=0)
                training_epochs=int(epochs_sum/train_set.shape[0])

                # read in the real test set
                with open(os.path.join(real_data_folder, 'data_sample_{}_{}_{}.pkl'.format('test', tic, D)), 'rb') as f:
                    real_test_dict = pickle.load(f)
                    # cast real_test to array
                    real_test=real_test_dict['data']
                    real_test=np.array(real_test)
                    test_time=real_test_dict['time']
                test_set=real_test

                # set the args for the model
                args.max_seq_len=train_set.shape[1]
                args.feature_dim=train_set.shape[2]
                args.label_dim=len(tics)

                # construct the train and test time
                # train set : (sample_number,seq_len,feature_dim)
                # train time: (sample_number,1) value is seq_len
                # T = train_set.shape[1]
                # train_time = np.array([T] * train_set.shape[0])
                # test_time = np.array([T] * test_set.shape[0])
                print('training_epochs',training_epochs)
                print('train_set.shape',train_set.shape)
                print('test_set.shape',test_set.shape)
                print('train_set preview',train_set[0:2,0:2,:])
                print('test_set preview', test_set[0:2, 0:2, :])
                pred_SMAPE, pred_norm_SMAPE, trainset_MSE = one_step_ahead_prediction(
                    (train_set, train_time),
                    (test_set, test_time),
                    args,
                    epochs=training_epochs,
                    model=args.prediction_model
                )
                pred_SMAPE_list.append(pred_SMAPE)
                pred_norm_SMAPE_list.append(pred_norm_SMAPE)
                trainset_MSE_list.append(trainset_MSE)
                avg_pred_SMAPE_list.append((pred_SMAPE+pred_norm_SMAPE)/2)

        if task=='benchmark_encoding':
            model+='_encoding'

        model_result = model_result.append({
            'pred_SMAPE': sum(pred_SMAPE_list) / len(pred_SMAPE_list),
            'pred_norm_SMAPE': sum(pred_norm_SMAPE_list) / len(pred_norm_SMAPE_list),
            'trainset_MSE': sum(trainset_MSE_list) / len(trainset_MSE_list),
            'avg_pred_SMAPE': sum(avg_pred_SMAPE_list)/len(avg_pred_SMAPE_list),
            'pred_SMAPE_std': np.std(pred_SMAPE_list),
            'pred_norm_SMAPE_std': np.std(pred_norm_SMAPE_list),
            'trainset_MSE_std': np.std(trainset_MSE_list),
            'avg_pred_SMAPE_std': np.std(avg_pred_SMAPE_list),
            'pred_SMAPE_detail': ','.join([str(i) for i in pred_SMAPE_list]),
            'pred_norm_SMAPE_detail': ','.join([str(i) for i in pred_norm_SMAPE_list]),
            'trainset_MSE_detail': ','.join([str(i) for i in trainset_MSE_list]),
            'avg_pred_SMAPE_detail': ','.join([str(i) for i in avg_pred_SMAPE_list]),
            # Repeated the pred_SMAPE_list as a placeholder, replace as needed
            'model_name': model,
            'experiment_time': datetime.datetime.now().strftime('%m%d%H%M')
        }, ignore_index=True)


        print(f'model_result of {model}',model_result)
        # if experiment_result_file does not exist, create it and write the header
        if not os.path.exists(experiment_result_file):
            model_result.to_csv(experiment_result_file, index=False)
        else:
            model_result.to_csv(experiment_result_file, mode='a', header=False, index=False)

