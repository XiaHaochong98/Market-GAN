from MarketGAN import MarketGAN
from utils.util import *
import torch
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
        "--is_train",
        type=str2bool,
        default=True)
    parser.add_argument(
        '--seed',
        default=-1,
        type=int)
    parser.add_argument(
        '--feat_pred_no',
        default=2,
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

    # Model Arguments
    parser.add_argument(
        '--emb_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--sup_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--gan_epochs',
        default=600,
        type=int)
    parser.add_argument(
        '--pretrain_epochs',
        default=200,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=20,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--dis_thresh',
        default=0.15,
        type=float)
    parser.add_argument(
        '--optimizer',
        choices=['adam'],
        default='adam',
        type=str)
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
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
        default=30,
        type=int)
    parser.add_argument(
        '--embedding_dim',
        default=20,
        type=int)
    parser.add_argument(
        "--conditional",
        type=str2bool,
        default=True)
    parser.add_argument(
        "--pretrain_model",
        type=str,
        default='')
    parser.add_argument(
        "--train_classifier",
        type=str2bool,
        default=False)
    parser.add_argument(
        "--load_supervisors",
        type=str2bool,
        default=False)
    parser.add_argument(
        "--pretrain_model_path",
        type=str,
        default='')
    parser.add_argument(
"--dynamic_weight",
        type=float,
        default=1.0)
    parser.add_argument(
        "--label_weight",
        type=float,
        default=1.0)
    parser.add_argument(
        "--generator_strengthen_factor",
        type=int,
        default=999999,
        help="The factor to strengthen the generator training, every x epoch step the generator will be trained for 1 more step")
    parser.add_argument(
        "--only_pretrain",
        type=str2bool,
        default=False,
        help="Only pretrain the condition classification model, no GAN training")
    parser.add_argument(
        "--PM_data_structure",
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '--history_length',
        type=int,
        default=30
    )
    parser.add_argument(
        '--differential_features',
        type=str2bool,
        default=True
    )
    parser.add_argument(
            '--use_TimesNet',
            type=str2bool,
            default=True
    )
    parser.add_argument(
            '--use_RNN',
            type=str2bool,
            default=False
    )
    parser.add_argument(
        '--task',
        type=str,
        default='full_experiment'
    )
    parser.add_argument(
        '--baseline_result_path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--baseline_real_data_path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--evalution_epochs',
        type=int,
        default=100
    )
    parser.add_argument(
        '--add_history',
        type=int,
        default=2
    )
    parser.add_argument(
        '--prepare_encoded_data',
        type=str2bool,
        default=False
    )
    parser.add_argument(
        '--latent_condtion_supervisor_model',
        type=str,
        default='TimesNet'
    )
    parser.add_argument(
        '--plot',
        type = str2bool,
        default = False
    )
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)


    Worker = MarketGAN()
    data_path, args = Worker.init(args)

    if args.task=='info_export':
        Worker.export_exp_info(data_path, args)
        print('Info exported successfully')
        exit(0)

    elif args.task=='data_provider':
        # prepare raw data samples for other baseline methods with ensemble methods by market dynamics and tics
        Worker.data_provider(data_path, args)
        print('Data rendered successfully')
        # exit the program
        exit(0)


    elif args.task=='baseline_evaluation':
        print('running baseline evaluation')
        args, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler, train_H, \
        test_H, train_Last_h, test_Last_h, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer, features = Worker.preprocess(
            data_path, args)
        args, model = Worker.load_model(args)
        real_data_folder= args.baseline_real_data_path
        result_path=args.baseline_result_path
        Worker.baseline_evaluation(args,model,result_path,real_data_folder)
        print('Baseline evaluation finished')
        exit(0)

    elif args.task == 'aggregation_baseline_evaluation':
        print('running aggregation_baseline_evaluation')
        args, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler, train_H, \
            test_H, train_Last_h, test_Last_h, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer, features = Worker.preprocess(
            data_path, args)
        args, model = Worker.load_model(args)
        real_data_folder = args.baseline_real_data_path
        result_path = args.baseline_result_path
        Worker.aggregation_baseline_evaluation(args, model, result_path, real_data_folder)
        print('aggregation_baseline_evaluation finished')
        exit(0)



    #preprocess
    args, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler, test_scaler, train_H, \
    test_H, train_Last_h, test_Last_h, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer, features = Worker.preprocess(data_path, args)

    # train
    if args.conditional:
        args, model = Worker.train(args, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L,
                                   train_scaler, test_scaler, train_H, test_H, train_Last_h, test_Last_h)
        # generation
        args, model, generated_train_X, generated_test_X, train_X, test_X, generated_data_as_train_rescaled, generated_data_as_test_rescaled, train_data_rescaled, test_data_rescaled, train_history_rescaled, test_history_rescaled, generated_time_as_test, generated_time_as_train= Worker.generation(
                args, model, train_X, test_X, train_time, test_time, train_D, test_D, train_L, test_L, train_scaler,
                test_scaler, train_H, test_H,train_Last_h,test_Last_h
            )

        # evaluation
        result_path=Worker.evaluate(args, model, dynamics_book, dynamics_tokenizer, label_book, tic_tokenizer,
                                    generated_train_X, generated_test_X, train_X, test_X,
                                    generated_data_as_train_rescaled, generated_data_as_test_rescaled, train_data_rescaled, test_data_rescaled,
                                    train_history_rescaled, test_history_rescaled,
                                    train_Last_h, test_Last_h,
                                    train_time, train_D, train_L, train_H, test_time, test_D, test_L, test_H, exp_name=args.exp, result_path=None,model_name=args.exp)


