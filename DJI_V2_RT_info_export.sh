#!/bin/bash
train=true
conditional=true
train_classifier=false
load_supervisors=true
use_TimesNet=true
use_RNN=true
export TZ="GMT-8"

# Experiment variables
exp="DJ30_V2_RT"

# Iteration variables
pretrain_epochs=1
emb_epochs=100
sup_epochs=100
gan_epochs=100

python main.py \
--device            cuda:2 \
--exp               $exp \
--is_train          $train \
--seed              42 \
--feat_pred_no      1 \
--max_seq_len       30 \
--train_rate        0.8 \
--emb_epochs        $emb_epochs \
--sup_epochs        $sup_epochs \
--gan_epochs        $gan_epochs \
--pretrain_epochs   $pretrain_epochs \
--batch_size        128 \
--hidden_dim        20 \
--num_layers        3 \
--dis_thresh        0.15 \
--optimizer         adam \
--learning_rate     1e-3 \
--data_path         ./data/DJI/DJI_50/tic/DJI_data_labeled_slice_and_merge_model_3dynamics_minlength50_quantile_labeling.csv \
--dynamic_dim       3 \
--label_dim         29 \
--embedding_dim     20 \
--conditional       $conditional \
--pretrain_model    TimesNet \
--train_classifier  $train_classifier \
--load_supervisors  $load_supervisors \
--pretrain_model_path ./output/DJI_pretrain_50 \
--dynamic_weight    1.0 \
--label_weight      1.0  \
--history_length    30 \
--generator_strengthen_factor 1 \
--use_TimesNet  $use_TimesNet \
--use_RNN $use_RNN \
--add_history 2 \
--latent_condtion_supervisor_model TimesNet \
--task info_export